"""HorseProno Quinté V4.4 — Winner Top 3.

V4.4 ne change jamais les 7 chevaux sélectionnés par V4.2/V4.3.
Elle réordonne uniquement les 7 chevaux sur les courses de PLAT afin d'améliorer
la présence du gagnant dans les 3 premières propositions.

Score PLAT retenu (recherche temporelle sur les prédictions walk-forward) :
    50% rang modèle Top5
    40% percentile LambdaMART
    10% forme/contexte

Hors PLAT, l'ordre V4.3 est conservé à l'identique.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from quinte_v43 import rank_quinte_v43

V44_CONFIG = {
    "mode": "WINNER_TOP3_PLAT",
    "flat_order_weights": {
        "top5_model": 0.50,
        "ranker": 0.40,
        "form_context": 0.10,
    },
    "shortlist_size": 7,
}


def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def rank_quinte_v44(
    history: pd.DataFrame,
    race: pd.DataFrame,
    artifact: dict[str, Any],
    ranker: Any,
    outsider_artifact: dict[str, Any],
) -> pd.DataFrame:
    result = rank_quinte_v43(history, race, artifact, ranker, outsider_artifact).copy()
    result["v44_original_rank"] = pd.to_numeric(result["quinte_rank"], errors="coerce")
    result["v44_winner_score"] = np.nan
    result["v44_mode"] = "V43_ORDER"

    if result.empty or str(result.iloc[0].get("discipline", "")).strip().upper() != "PLAT":
        return result

    selected = result[result["selected_top7"].astype(bool)].copy()
    remaining = result[~result["selected_top7"].astype(bool)].copy()
    if selected.empty:
        return result

    top5 = _num(selected["model_rankpct"])
    ranker_pct = _num(selected["ranker_percentile"])
    form = _num(selected.get("form_pct", pd.Series(0.0, index=selected.index)))
    consistency = _num(selected.get("consistency_pct", pd.Series(0.0, index=selected.index)))
    context = _num(selected["context_score"])
    form_context = (form + consistency + context) / 3.0

    w = V44_CONFIG["flat_order_weights"]
    selected["v44_winner_score"] = (
        w["top5_model"] * top5
        + w["ranker"] * ranker_pct
        + w["form_context"] * form_context
    )
    selected = selected.sort_values(
        ["v44_winner_score", "top5_probability", "ranker_score", "horse_number"],
        ascending=[False, False, False, True],
    )

    # Les chevaux hors Top7 gardent l'ordre V4.3 : V4.4 ne touche qu'à l'ordre du Top7.
    out = pd.concat([selected, remaining], axis=0).reset_index(drop=True)
    out["quinte_rank"] = np.arange(1, len(out) + 1)
    out["selected_top7"] = out["quinte_rank"].le(int(V44_CONFIG["shortlist_size"]))
    out["v44_mode"] = "WINNER_TOP3_PLAT"
    return out
