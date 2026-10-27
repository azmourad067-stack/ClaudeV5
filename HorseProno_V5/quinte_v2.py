"""HorseProno Quinté V2.

V2 est un modèle de *shortlist* :
- noyau de 6 chevaux = probabilité de marché normalisée dans la course ;
- 1 challenger = meilleur cheval hors noyau selon le modèle Top-5 calibré ;
- LambdaMART est conservé comme challenger offline, pas forcé en production.

Le but n'est pas de présenter l'edge comme certain : V2 optimise la couverture du Top 5
réel dans un Top 7, conformément au walk-forward historique.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quinte_candidate import prepare

RELATIVE_SOURCES = [
    "market_prob",
    "form",
    "consistency",
    "horse_name_top5",
    "horse_name_win",
    "jockey_top5",
    "jockey_win",
    "trainer_top5",
    "trainer_win",
]


def add_relative_features(base: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des variables relatives à la course, sans utiliser le résultat.

    `race` et `base` doivent être alignés ligne à ligne.
    Les rangs percentiles sont orientés pour que 1 = meilleur de la course et 0 = pire.
    """
    if len(base) != len(race):
        raise ValueError("base et race doivent avoir le même nombre de lignes")
    out = base.copy().reset_index(drop=True)
    race = race.reset_index(drop=True)
    if "race_id" not in race.columns:
        raise ValueError("race_id absent")

    group = race["race_id"].astype(str)
    for col in RELATIVE_SOURCES:
        if col not in out.columns:
            raise ValueError(f"Variable relative absente: {col}")
        v = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[f"{col}_rel"] = v - v.groupby(group).transform("mean")
        r = v.groupby(group).rank(method="average", ascending=False)
        n = v.groupby(group).transform("size").astype(float)
        out[f"{col}_rankpct"] = np.where(n > 1, 1.0 - (r - 1.0) / (n - 1.0), 1.0)

    out["market_form_gap"] = out["market_prob_rankpct"] - out["form_rankpct"]
    out["market_horse_gap"] = out["market_prob_rankpct"] - out["horse_name_top5_rankpct"]
    return out.astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def predict_top5_probability(artifact: dict[str, Any], base: pd.DataFrame) -> np.ndarray:
    lg = artifact["top5_logistic"]
    cols = lg["features"]
    x = base[cols].to_numpy(dtype=float)
    mean = np.asarray(lg["mean"], dtype=float)
    scale = np.asarray(lg["scale"], dtype=float)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    coef = np.asarray(lg["coef"], dtype=float)
    raw = _sigmoid(((x - mean) / scale) @ coef + float(lg["intercept"]))

    cal = lg.get("platt_calibrator") or {}
    if cal:
        raw = _sigmoid(float(cal["coef"]) * _logit(raw) + float(cal["intercept"]))
    return raw


def _normalise_ranker_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if len(score) <= 1:
        return np.ones_like(score)
    order = pd.Series(score).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (order - 1.0) / max(1.0, len(score) - 1.0)


def rank_quinte_v2(
    history: pd.DataFrame,
    race: pd.DataFrame,
    artifact: dict[str, Any],
    ranker: Any | None = None,
) -> pd.DataFrame:
    """Classe une course et produit le Top7 V2.

    Les six premiers du marché normalisé forment le noyau. Le meilleur outsider selon
    la probabilité Top-5 calibrée devient le challenger n°7. LambdaMART, s'il est fourni,
    n'intervient pas dans la sélection de production et sert uniquement de diagnostic.
    """
    race = race.copy().reset_index(drop=True)
    base = prepare(history, race).reset_index(drop=True)
    full = add_relative_features(base, race)

    top5_prob = predict_top5_probability(artifact, base)
    market_prob = pd.to_numeric(base["market_prob"], errors="coerce").fillna(0.0).to_numpy()

    result = race.copy()
    result["market_probability"] = market_prob
    result["top5_probability"] = top5_prob
    result["market_rank"] = (
        pd.Series(market_prob).rank(method="first", ascending=False).astype(int).to_numpy()
    )
    result["model_top5_rank"] = (
        pd.Series(top5_prob).rank(method="first", ascending=False).astype(int).to_numpy()
    )

    # LambdaMART reste disponible uniquement pour audit/diagnostic.
    if ranker is not None and artifact.get("ranker", {}).get("features"):
        missing = [c for c in artifact["ranker"]["features"] if c not in full.columns]
        if missing:
            raise RuntimeError("Variables ranker absentes: " + ", ".join(missing))
        ranker_score = np.asarray(ranker.predict(full[artifact["ranker"]["features"]]), dtype=float)
        result["ranker_score"] = ranker_score
        result["ranker_percentile"] = _normalise_ranker_score(ranker_score)
        result["ranker_rank"] = pd.Series(ranker_score).rank(method="first", ascending=False).astype(int).to_numpy()
    else:
        result["ranker_score"] = np.nan
        result["ranker_percentile"] = np.nan
        result["ranker_rank"] = np.nan

    core_size = int(artifact["strategy"]["market_core_size"])
    challenger_count = int(artifact["strategy"]["model_challengers"])
    core_idx = (
        result.sort_values(["market_probability", "horse_number"], ascending=[False, True])
        .head(core_size)
        .index.tolist()
    )
    outsider = result.drop(index=core_idx).sort_values(
        ["top5_probability", "market_probability", "horse_number"],
        ascending=[False, False, True],
    )
    challenger_idx = outsider.head(challenger_count).index.tolist()

    core_order = result.loc[core_idx].sort_values(
        ["market_probability", "horse_number"], ascending=[False, True]
    ).index.tolist()
    challenger_order = result.loc[challenger_idx].sort_values(
        ["top5_probability", "market_probability", "horse_number"], ascending=[False, False, True]
    ).index.tolist()
    remaining = result.drop(index=core_order + challenger_order).sort_values(
        ["top5_probability", "market_probability", "horse_number"],
        ascending=[False, False, True],
    ).index.tolist()

    ordered_idx = core_order + challenger_order + remaining
    result["shortlist_role"] = "HORS_TOP7"
    result.loc[core_order, "shortlist_role"] = "NOYAU_MARCHE"
    result.loc[challenger_order, "shortlist_role"] = "CHALLENGER_MODELE"

    result = result.loc[ordered_idx].reset_index(drop=True)
    result["quinte_rank"] = np.arange(1, len(result) + 1)
    result["selected_top7"] = result["quinte_rank"].le(7)
    return result


def load_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
