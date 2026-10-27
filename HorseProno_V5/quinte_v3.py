"""HorseProno Quinté V3 candidate — 4 noyaux + 3 challengers.

Architecture retenue sur le front de Pareto historique (58 courses walk-forward):
- noyau de 4 = consensus 10% marché / 30% logit Top5 / 50% LambdaMART / 10% contexte,
  avec au maximum 2 chevaux issus du Top4 marché ;
- challenger 1 = meilleur LambdaMART hors noyau ;
- challenger 2 = meilleur marché restant (stabilisateur) ;
- challenger 3 = outsider profond (rang marché >= 10 si disponible) maximisant un score d'edge.

Le moteur n'utilise jamais l'arrivée de la course courante pour construire la sélection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quinte_candidate import prepare
from quinte_v2 import add_relative_features, predict_top5_probability

V3_CONFIG = {
    "core_weights": {"market": 0.10, "top5_logit": 0.30, "ranker": 0.50, "context": 0.10},
    "core_market_top4_cap": 2,
    "deep_market_rank_min": 10,
    "deep_edge_weights": {"top5_logit": 0.55, "ranker": 0.25, "context": 0.20, "market_penalty": 0.40},
}


def _rankpct(values: pd.Series | np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    n = len(s)
    if n <= 1:
        return np.ones(n, dtype=float)
    r = s.rank(method="first", ascending=False).to_numpy(dtype=float)
    return 1.0 - (r - 1.0) / float(n - 1)


def _context_score(full: pd.DataFrame) -> np.ndarray:
    entity = (
        full["horse_name_top5_rankpct"].to_numpy(dtype=float)
        + full["jockey_top5_rankpct"].to_numpy(dtype=float)
        + full["trainer_top5_rankpct"].to_numpy(dtype=float)
    ) / 3.0
    return (
        0.45 * full["form_rankpct"].to_numpy(dtype=float)
        + 0.15 * full["consistency_rankpct"].to_numpy(dtype=float)
        + 0.40 * entity
    )


def load_ranker(model_file: str | Path):
    """Charge le ranker V3.

    LightGBM est utilisé lorsqu'il est disponible. Sur Streamlit Cloud, un lecteur
    Python pur du même fichier modèle prend automatiquement le relais afin que
    l'application ne dépende pas du binaire LightGBM.
    """
    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(model_file))
    except (ImportError, ModuleNotFoundError, OSError):
        from portable_lgbm import PortableBooster
        return PortableBooster(model_file)


def rank_quinte_v3(
    history: pd.DataFrame,
    race: pd.DataFrame,
    artifact: dict[str, Any],
    ranker: Any,
) -> pd.DataFrame:
    if ranker is None:
        raise ValueError("V3 nécessite le modèle LambdaMART (ranker).")

    race = race.copy().reset_index(drop=True)
    base = prepare(history, race).reset_index(drop=True)
    full = add_relative_features(base, race)

    market_prob = pd.to_numeric(base["market_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    top5_prob = predict_top5_probability(artifact, base)

    features = artifact.get("ranker", {}).get("features", [])
    missing = [c for c in features if c not in full.columns]
    if missing:
        raise RuntimeError("Variables LambdaMART absentes: " + ", ".join(missing))
    ranker_score = np.asarray(ranker.predict(full[features]), dtype=float)

    market_pct = _rankpct(market_prob)
    logit_pct = _rankpct(top5_prob)
    ranker_pct = _rankpct(ranker_score)
    context_pct = _context_score(full)

    result = race.copy()
    result["market_probability"] = market_prob
    result["top5_probability"] = top5_prob
    result["ranker_score"] = ranker_score
    result["market_rankpct"] = market_pct
    result["model_rankpct"] = logit_pct
    result["ranker_percentile"] = ranker_pct
    result["context_score"] = context_pct
    result["market_rank"] = pd.Series(market_prob).rank(method="first", ascending=False).astype(int).to_numpy()
    result["model_top5_rank"] = pd.Series(top5_prob).rank(method="first", ascending=False).astype(int).to_numpy()
    result["ranker_rank"] = pd.Series(ranker_score).rank(method="first", ascending=False).astype(int).to_numpy()

    # Diagnostics pré-course exposés pour le détecteur outsider V4.3.
    # Ces colonnes sont toutes construites sans utiliser l'arrivée de la course courante.
    result["form_pct"] = pd.to_numeric(full["form_rankpct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    result["consistency_pct"] = pd.to_numeric(full["consistency_rankpct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    entity_pct = (
        pd.to_numeric(full["horse_name_top5_rankpct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        + pd.to_numeric(full["jockey_top5_rankpct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        + pd.to_numeric(full["trainer_top5_rankpct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ) / 3.0
    result["entity_pct"] = entity_pct
    result["model_cons_pct"] = 0.55 * logit_pct + 0.30 * ranker_pct + 0.15 * context_pct
    for _col in ["horse_name_top5", "horse_name_win", "jockey_top5", "jockey_win", "trainer_top5", "trainer_win"]:
        result[_col] = pd.to_numeric(base[_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    result["core_score"] = (
        V3_CONFIG["core_weights"]["market"] * market_pct
        + V3_CONFIG["core_weights"]["top5_logit"] * logit_pct
        + V3_CONFIG["core_weights"]["ranker"] * ranker_pct
        + V3_CONFIG["core_weights"]["context"] * context_pct
    )

    # 4 noyaux, avec cap de deux chevaux appartenant au Top4 marché.
    ordered_core = result.sort_values(
        ["core_score", "top5_probability", "ranker_score", "market_probability", "horse_number"],
        ascending=[False, False, False, False, True],
    )
    core_idx: list[int] = []
    market_top4_count = 0
    cap = int(V3_CONFIG["core_market_top4_cap"])
    for idx, row in ordered_core.iterrows():
        is_market_top4 = int(row["market_rank"]) <= 4
        if is_market_top4 and market_top4_count >= cap:
            continue
        core_idx.append(int(idx))
        market_top4_count += int(is_market_top4)
        if len(core_idx) == 4:
            break
    if len(core_idx) < 4:
        core_idx = ordered_core.head(4).index.astype(int).tolist()

    used = set(core_idx)

    # Challenger 1: meilleur LambdaMART restant.
    remaining = result.drop(index=list(used))
    c1 = int(
        remaining.sort_values(
            ["ranker_score", "top5_probability", "market_probability", "horse_number"],
            ascending=[False, False, False, True],
        ).index[0]
    )
    used.add(c1)

    # Challenger 2: meilleur cheval du marché restant, rôle stabilisateur.
    remaining = result.drop(index=list(used))
    c2 = int(
        remaining.sort_values(
            ["market_probability", "top5_probability", "horse_number"],
            ascending=[False, False, True],
        ).index[0]
    )
    used.add(c2)

    # Challenger 3: outsider profond; s'il n'y a pas de rang 10+, on élargit au reste.
    remaining = result.drop(index=list(used)).copy()
    deep = remaining[remaining["market_rank"] >= int(V3_CONFIG["deep_market_rank_min"])].copy()
    if deep.empty:
        deep = remaining.copy()
    deep["deep_edge_score"] = (
        V3_CONFIG["deep_edge_weights"]["top5_logit"] * deep["model_rankpct"]
        + V3_CONFIG["deep_edge_weights"]["ranker"] * deep["ranker_percentile"]
        + V3_CONFIG["deep_edge_weights"]["context"] * deep["context_score"]
        - V3_CONFIG["deep_edge_weights"]["market_penalty"] * deep["market_rankpct"]
    )
    c3 = int(
        deep.sort_values(
            ["deep_edge_score", "top5_probability", "context_score", "horse_number"],
            ascending=[False, False, False, True],
        ).index[0]
    )

    challenger_idx = [c1, c2, c3]
    selected = core_idx + challenger_idx

    # Ordre final: noyau puis challengers, sans modifier leur rôle.
    remaining_idx = result.drop(index=selected).sort_values(
        ["core_score", "top5_probability", "ranker_score", "market_probability", "horse_number"],
        ascending=[False, False, False, False, True],
    ).index.astype(int).tolist()

    result["shortlist_role"] = "HORS_TOP7"
    result.loc[core_idx, "shortlist_role"] = "NOYAU_V3"
    result.loc[c1, "shortlist_role"] = "CHALLENGER_RANKER"
    result.loc[c2, "shortlist_role"] = "CHALLENGER_STABILISATEUR"
    result.loc[c3, "shortlist_role"] = "CHALLENGER_EDGE_PROFOND"

    result = result.loc[selected + remaining_idx].reset_index(drop=True)
    result["quinte_rank"] = np.arange(1, len(result) + 1)
    result["selected_top7"] = result["quinte_rank"].le(7)
    return result
