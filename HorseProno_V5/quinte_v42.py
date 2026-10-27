"""HorseProno Quinté V4.2 — PLAT contextuel + V4 générale hors PLAT.

La V4.2 conserve l'architecture de fusion V2/V3. En PLAT, elle adapte uniquement
la pénalisation du marché selon l'écart de poids pré-course (proxy de structure du
handicap) :
- écart <= 6 kg        : marché = -0.80
- 6 < écart <= 8 kg   : marché = -1.00
- écart > 8 kg        : marché = -0.50

Les autres composantes PLAT restent celles de V4.1 :
0.60 modèle Top5 + 0.20 LambdaMART + 0.10 contexte.

Hors PLAT, la recette V4 générale est conservée :
0.60 modèle Top5 + 0.25 LambdaMART + 0.15 contexte - 0.40 marché.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from quinte_v3 import rank_quinte_v3

V42_CONFIG = {
    "shortlist_size": 7,
    "general_edge_weights": {
        "top5_model": 0.60,
        "ranker": 0.25,
        "context": 0.15,
        "market": -0.40,
    },
    "flat_base_weights": {
        "top5_model": 0.60,
        "ranker": 0.20,
        "context": 0.10,
    },
    "flat_market_by_weight_spread": {
        "le_6kg": -0.80,
        "gt_6_le_8kg": -1.00,
        "gt_8kg": -0.50,
        "fallback": -0.80,
    },
}


def _horse_key(v: Any) -> int:
    return int(float(v))


def _is_flat(discipline: Any) -> bool:
    return str(discipline or "").strip().upper() == "PLAT"


def _flat_market_weight(result: pd.DataFrame) -> tuple[float, float | None, str]:
    w = pd.to_numeric(result.get("weight"), errors="coerce").dropna()
    if len(w) < 2:
        return float(V42_CONFIG["flat_market_by_weight_spread"]["fallback"]), None, "POIDS_INCOMPLETS"
    spread = float(w.max() - w.min())
    cfg = V42_CONFIG["flat_market_by_weight_spread"]
    if spread <= 6.0:
        return float(cfg["le_6kg"]), spread, "ECART_POIDS_LE_6"
    if spread <= 8.0:
        return float(cfg["gt_6_le_8kg"]), spread, "ECART_POIDS_6_8"
    return float(cfg["gt_8kg"]), spread, "ECART_POIDS_GT_8"


def rank_quinte_v42(
    history: pd.DataFrame,
    race: pd.DataFrame,
    artifact: dict[str, Any],
    ranker: Any,
) -> pd.DataFrame:
    if ranker is None:
        raise ValueError("V4.2 nécessite le ranker LambdaMART/portable utilisé par V3.")

    base = rank_quinte_v3(history, race, artifact, ranker=ranker).copy()
    base["horse_number"] = base["horse_number"].map(_horse_key)

    # Reconstruit exactement la shortlist V2 6+1 à partir des sorties déjà calculées.
    core_size = int(artifact["strategy"]["market_core_size"])
    challenger_count = int(artifact["strategy"]["model_challengers"])
    v2_core = base.sort_values(
        ["market_probability", "horse_number"], ascending=[False, True]
    ).head(core_size)
    v2_core_nums = v2_core["horse_number"].astype(int).tolist()
    v2_out = base.loc[~base["horse_number"].isin(v2_core_nums)].sort_values(
        ["top5_probability", "market_probability", "horse_number"],
        ascending=[False, False, True],
    )
    v2_top = v2_core_nums + v2_out.head(challenger_count)["horse_number"].astype(int).tolist()
    v3_top = base.loc[base["selected_top7"], "horse_number"].astype(int).tolist()
    v2_set, v3_set = set(v2_top), set(v3_top)

    common = [h for h in v3_top if h in v2_set]
    union: list[int] = []
    for h in v2_top + v3_top:
        if h not in union:
            union.append(h)

    result = base.copy()
    result["in_v2_top7"] = result["horse_number"].isin(v2_set)
    result["in_v3_top7"] = result["horse_number"].isin(v3_set)
    result["v2_v3_consensus"] = result["in_v2_top7"] & result["in_v3_top7"]

    discipline = str(result.iloc[0].get("discipline", "") or "").strip().upper()
    flat_mode = _is_flat(discipline)

    if flat_mode:
        market_weight, weight_spread, context_bucket = _flat_market_weight(result)
        w = V42_CONFIG["flat_base_weights"]
        result["v42_edge_score"] = (
            w["top5_model"] * pd.to_numeric(result["model_rankpct"], errors="coerce").fillna(0.0)
            + w["ranker"] * pd.to_numeric(result["ranker_percentile"], errors="coerce").fillna(0.0)
            + w["context"] * pd.to_numeric(result["context_score"], errors="coerce").fillna(0.0)
            + market_weight * pd.to_numeric(result["market_rankpct"], errors="coerce").fillna(0.0)
        )
        result["v42_mode"] = "PLAT_CONTEXTUEL"
        result["v42_weight_spread"] = np.nan if weight_spread is None else weight_spread
        result["v42_market_weight"] = market_weight
        result["v42_context_bucket"] = context_bucket
    else:
        w = V42_CONFIG["general_edge_weights"]
        result["v42_edge_score"] = (
            w["top5_model"] * pd.to_numeric(result["model_rankpct"], errors="coerce").fillna(0.0)
            + w["ranker"] * pd.to_numeric(result["ranker_percentile"], errors="coerce").fillna(0.0)
            + w["context"] * pd.to_numeric(result["context_score"], errors="coerce").fillna(0.0)
            + w["market"] * pd.to_numeric(result["market_rankpct"], errors="coerce").fillna(0.0)
        )
        result["v42_mode"] = "V4_GENERAL"
        result["v42_weight_spread"] = np.nan
        result["v42_market_weight"] = float(w["market"])
        result["v42_context_bucket"] = "HORS_PLAT"

    # Aliases utiles aux exports historiques.
    result["v4_edge_score"] = result["v42_edge_score"]

    available = set(result["horse_number"].astype(int).tolist())
    common = [h for h in common if h in available]
    disagreement = [h for h in union if h not in set(common) and h in available]

    slots = int(V42_CONFIG["shortlist_size"]) - len(common)
    if slots < 0:
        common = common[: int(V42_CONFIG["shortlist_size"])]
        slots = 0

    disagreement_df = (
        result.loc[result["horse_number"].isin(disagreement)].copy()
        if disagreement else result.iloc[0:0].copy()
    )
    if not disagreement_df.empty:
        disagreement_df = disagreement_df.sort_values(
            ["v42_edge_score", "top5_probability", "ranker_score", "context_score", "horse_number"],
            ascending=[False, False, False, False, True],
        )
    chosen = disagreement_df["horse_number"].astype(int).head(slots).tolist()

    selected = common + chosen
    if len(selected) < int(V42_CONFIG["shortlist_size"]):
        fallback = result.loc[~result["horse_number"].isin(selected)].sort_values(
            ["v42_edge_score", "top5_probability", "ranker_score", "horse_number"],
            ascending=[False, False, False, True],
        )
        selected += fallback["horse_number"].astype(int).head(
            int(V42_CONFIG["shortlist_size"]) - len(selected)
        ).tolist()

    common_df = result.loc[result["horse_number"].isin(common)].copy() if common else result.iloc[0:0].copy()
    if not common_df.empty:
        common_df["_display_score"] = (
            0.45 * common_df["model_rankpct"]
            + 0.30 * common_df["ranker_percentile"]
            + 0.15 * common_df["market_rankpct"]
            + 0.10 * common_df["context_score"]
        )
        common_df = common_df.sort_values(
            ["_display_score", "top5_probability", "horse_number"],
            ascending=[False, False, True],
        )

    chosen_df = result.loc[result["horse_number"].isin(chosen)].copy() if chosen else result.iloc[0:0].copy()
    if not chosen_df.empty:
        chosen_df = chosen_df.sort_values(
            ["v42_edge_score", "top5_probability", "horse_number"],
            ascending=[False, False, True],
        )

    selected_order = common_df["horse_number"].astype(int).tolist() + chosen_df["horse_number"].astype(int).tolist()
    for h in selected:
        if h not in selected_order:
            selected_order.append(h)

    remaining = result.loc[~result["horse_number"].isin(selected_order)].sort_values(
        ["v42_edge_score", "top5_probability", "ranker_score", "market_probability", "horse_number"],
        ascending=[False, False, False, False, True],
    )["horse_number"].astype(int).tolist()

    result["shortlist_role"] = "HORS_TOP7"
    result.loc[result["horse_number"].isin(common), "shortlist_role"] = "CONSENSUS_V2_V3"
    result.loc[result["horse_number"].isin(chosen), "shortlist_role"] = "ARBITRAGE_EDGE_V42"

    ordered = selected_order + remaining
    result = result.set_index("horse_number", drop=False).loc[ordered].reset_index(drop=True)
    result["quinte_rank"] = np.arange(1, len(result) + 1)
    result["selected_top7"] = result["quinte_rank"].le(int(V42_CONFIG["shortlist_size"]))
    result["v42_consensus_count"] = len(common)
    result["v42_disagreement_slots"] = int(V42_CONFIG["shortlist_size"]) - len(common)
    return result
