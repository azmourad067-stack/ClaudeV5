"""HorseProno Quinté V4.5 — Market Dynamics / projection de cote T-2.

Objectif
--------
V4.5 conserve exactement les 7 chevaux de V4.2/V4.3. Sur le PLAT, elle part de
l'ordre Winner Top3 de V4.4 et ajoute un faible ajustement lié à la dynamique de
marché pré-course :

- momentum : direction probable de la cote jusqu'à T-2 ;
- volatilité : instabilité des mouvements observés ;
- confiance : qualité du signal selon le nombre de snapshots et leur régularité.

Aucune cote post-course n'est utilisée dans la prédiction. Quand il n'existe pas
encore plusieurs snapshots live, le différentiel PMU direct / référence sert de
proxy conservateur. Dès que plusieurs snapshots sont présents dans la session,
une pente sur log(cote) prend le relais.

Cette version est volontairement expérimentale : le coefficient maximal est borné
à +/- 0.10 du score Winner V4.4 afin de ne pas laisser le marché tardif écraser les
signaux sportifs du modèle.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
import math
import numpy as np
import pandas as pd

V45_CONFIG = {
    "target_minutes_before_start": 2.0,
    "max_winner_adjustment": 0.10,
    "reference_extrapolation": 0.35,
    "live_slope_damping": 0.55,
    "max_abs_predicted_log_move": 0.55,
    "reference_only_confidence": 0.35,
    "two_snapshot_confidence": 0.65,
    "three_plus_snapshot_confidence": 0.90,
    "volatility_penalty": 2.0,
}


def _f(v: Any) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _parse_ts(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _clip_move(x: float) -> float:
    m = float(V45_CONFIG["max_abs_predicted_log_move"])
    return float(np.clip(x, -m, m))


def _live_projection(
    current_odds: float,
    minutes_to_start: float | None,
    observations: list[dict[str, Any]],
) -> tuple[float | None, float, float, str]:
    """Retourne (predicted_log_move, volatility, base_confidence, source)."""
    clean: list[tuple[datetime, float]] = []
    for obs in observations or []:
        ts = _parse_ts(obs.get("captured_at"))
        odds = _f(obs.get("odds"))
        if ts is not None and odds is not None and odds > 1:
            clean.append((ts, odds))
    clean.sort(key=lambda x: x[0])

    # Déduplique les timestamps identiques et garde au maximum les 5 plus récents.
    dedup: dict[str, tuple[datetime, float]] = {}
    for ts, odds in clean:
        dedup[ts.isoformat()] = (ts, odds)
    clean = list(dedup.values())[-5:]

    if len(clean) < 2:
        return None, 0.0, 0.0, "NO_LIVE_SLOPE"

    t0 = clean[0][0]
    x = np.array([(ts - t0).total_seconds() / 60.0 for ts, _ in clean], dtype=float)
    y = np.log(np.array([odds for _, odds in clean], dtype=float))
    if np.ptp(x) < 0.5:
        return None, 0.0, 0.0, "NO_LIVE_SLOPE"

    # Pente robuste simple : régression linéaire sur les snapshots récents.
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    step_returns = np.diff(y)
    volatility = float(np.std(step_returns, ddof=0)) if len(step_returns) else float(np.std(residual, ddof=0))

    mts = _f(minutes_to_start)
    if mts is None:
        horizon = 0.0
    else:
        horizon = max(0.0, mts - float(V45_CONFIG["target_minutes_before_start"]))
    predicted_move = _clip_move(float(slope) * horizon * float(V45_CONFIG["live_slope_damping"]))
    base_conf = (
        float(V45_CONFIG["two_snapshot_confidence"])
        if len(clean) == 2
        else float(V45_CONFIG["three_plus_snapshot_confidence"])
    )
    return predicted_move, volatility, base_conf, "LIVE_SLOPE"


def _reference_projection(direct: float | None, reference: float | None) -> tuple[float | None, float, float, str]:
    if direct is None or reference is None or direct <= 1 or reference <= 1:
        return None, 0.0, 0.0, "NO_REFERENCE"
    # Si la cote directe a déjà baissé sous la référence, on projette une petite
    # continuation de ce mouvement. Le facteur 0.35 est volontairement prudent
    # tant que nous n'avons pas encore un historique T30/T2 suffisamment fourni.
    observed_log_gap = math.log(direct / reference)
    predicted_move = _clip_move(float(V45_CONFIG["reference_extrapolation"]) * observed_log_gap)
    volatility = abs(observed_log_gap) * 0.50
    return predicted_move, volatility, float(V45_CONFIG["reference_only_confidence"]), "DIRECT_REFERENCE"


def apply_v45_market_dynamics(
    ranked_v44: pd.DataFrame,
    snapshot_history: dict[int, list[dict[str, Any]]] | None = None,
    minutes_to_start: float | None = None,
) -> pd.DataFrame:
    """Ajoute la couche V4.5 et réordonne uniquement le Top7 PLAT.

    `snapshot_history` est un mapping horse_number -> observations live de la
    session courante. Le Top7 reste strictement identique à V4.4 ; seul l'ordre
    de ces sept chevaux peut changer.
    """
    out = ranked_v44.copy().reset_index(drop=True)
    out["v45_original_rank"] = pd.to_numeric(out.get("quinte_rank"), errors="coerce")
    out["v45_projected_odds_t2"] = np.nan
    out["v45_predicted_log_move"] = 0.0
    out["v45_projected_change_pct"] = 0.0
    out["v45_volatility"] = 0.0
    out["v45_dynamics_confidence"] = 0.0
    out["v45_dynamics_score"] = 0.0
    out["v45_adjustment"] = 0.0
    out["v45_winner_score"] = pd.to_numeric(out.get("v44_winner_score"), errors="coerce")
    out["v45_signal_source"] = "OFF"
    out["v45_mode"] = "V44_ORDER"

    if out.empty or str(out.iloc[0].get("discipline", "")).strip().upper() != "PLAT":
        return out

    histories = snapshot_history or {}
    raw_momentum: list[float] = []
    vols: list[float] = []
    base_conf: list[float] = []
    sources: list[str] = []
    projected: list[float | None] = []
    pred_moves: list[float] = []

    for row in out.itertuples(index=False):
        current = _f(getattr(row, "odds_direct", None)) or _f(getattr(row, "odds", None))
        ref = _f(getattr(row, "odds_reference", None))
        hnum = int(getattr(row, "horse_number"))
        obs = histories.get(hnum, [])

        live_move, live_vol, live_conf, live_source = _live_projection(current or 0.0, minutes_to_start, obs)
        if live_move is not None:
            move, vol, conf, src = live_move, live_vol, live_conf, live_source
        else:
            move, vol, conf, src = _reference_projection(current, ref)

        if move is None or current is None or current <= 1:
            move = 0.0
            proj = current
            conf = 0.0
            src = "NO_SIGNAL"
        else:
            proj = max(1.01, current * math.exp(move))

        # momentum positif = cote projetée à la baisse (signal favorable marché).
        momentum = -float(move)
        raw_momentum.append(momentum)
        vols.append(float(max(0.0, vol)))
        base_conf.append(float(np.clip(conf, 0.0, 1.0)))
        sources.append(src)
        projected.append(proj)
        pred_moves.append(float(move))

    mom = np.asarray(raw_momentum, dtype=float)
    vol = np.asarray(vols, dtype=float)

    def zscore(a: np.ndarray) -> np.ndarray:
        sd = float(np.std(a, ddof=0))
        if sd < 1e-9:
            return np.zeros_like(a)
        return (a - float(np.mean(a))) / sd

    mom_z = zscore(mom)
    vol_z = zscore(vol)
    conf_arr = np.asarray(base_conf, dtype=float) * np.exp(-float(V45_CONFIG["volatility_penalty"]) * vol)
    conf_arr = np.clip(conf_arr, 0.0, 1.0)
    # + : contraction projetée et mouvement régulier ; - : dérive / forte volatilité.
    dyn = np.tanh(mom_z - 0.30 * np.maximum(vol_z, -1.0)) * conf_arr
    dyn = np.clip(dyn, -1.0, 1.0)

    out["v45_projected_odds_t2"] = projected
    out["v45_predicted_log_move"] = pred_moves
    current_series = pd.to_numeric(out.get("odds_direct", out.get("odds")), errors="coerce")
    if "odds_direct" in out.columns:
        current_series = pd.to_numeric(out["odds_direct"], errors="coerce").fillna(pd.to_numeric(out["odds"], errors="coerce"))
    out["v45_projected_change_pct"] = 100.0 * (pd.to_numeric(out["v45_projected_odds_t2"], errors="coerce") / current_series - 1.0)
    out["v45_volatility"] = vol
    out["v45_dynamics_confidence"] = conf_arr
    out["v45_dynamics_score"] = dyn
    out["v45_signal_source"] = sources

    max_adj = float(V45_CONFIG["max_winner_adjustment"])
    out["v45_adjustment"] = max_adj * out["v45_dynamics_score"]
    base_score = pd.to_numeric(out["v44_winner_score"], errors="coerce")
    # Hors Top7 le score est laissé inchangé et ces chevaux ne sont pas réordonnés.
    out["v45_winner_score"] = base_score.fillna(0.0) + out["v45_adjustment"]

    selected = out[out["selected_top7"].astype(bool)].copy()
    remaining = out[~out["selected_top7"].astype(bool)].copy()
    selected = selected.sort_values(
        ["v45_winner_score", "v44_winner_score", "top5_probability", "horse_number"],
        ascending=[False, False, False, True],
    )
    final = pd.concat([selected, remaining], ignore_index=True)
    final["quinte_rank"] = np.arange(1, len(final) + 1)
    final["selected_top7"] = final["quinte_rank"].le(7)
    final["v45_mode"] = "MARKET_DYNAMICS_PLAT"
    return final
