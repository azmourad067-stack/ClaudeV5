"""Outils communs d'évaluation (recherche). Aucune donnée d'arrivée n'entre dans les features."""
from __future__ import annotations
import sys, json, ast
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_history() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "validated_history.csv", low_memory=False)
    h["race_date"] = pd.to_datetime(h["race_date"]).dt.normalize()
    return h


def load_forward(odds_source: str = "snap") -> pd.DataFrame:
    """41 Quintés du 12/09 au 22/10/2026, jamais vus par V2..V4.5.

    odds_source='snap' : cote PMU capturée ~T-20 (condition réelle d'utilisation)
    odds_source='db'   : participants.odds (horodatage d'ingestion inconnu)
    """
    df = pd.read_csv(ROOT / "data" / "forward_quinte_raw.csv", keep_default_na=False)
    df = df[df.is_non_runner.astype(str).str.lower().isin(["f", "false"])].copy()
    num = lambda c: pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    out = pd.DataFrame({
        "race_id": "R" + df.meeting_number.astype(str) + "C" + df.race_number.astype(str) + "_" + df.race_date,
        "race_date": pd.to_datetime(df.race_date),
        "discipline": df.discipline, "hippodrome": df.hippodrome, "distance": num("distance"),
        "terrain": df.terrain, "horse_number": num("horse_number").astype(int),
        "horse_name": df.horse_name, "jockey": df.jockey, "trainer": df.trainer,
        "odds_db": num("odds"), "odds_snap": num("odds_snap"),
        "draw": num("draw"), "weight": num("weight"), "age": num("age"), "sex": df.sex,
        "recent_form": df.recent_form, "is_non_runner": False,
    })
    out["odds"] = out["odds_snap"] if odds_source == "snap" else out["odds_db"]
    if odds_source == "snap":  # repli sur la cote DB si le snapshot manque (1 %)
        out["odds"] = out["odds"].fillna(out["odds_db"])
    out["field_size"] = out.groupby("race_id").horse_number.transform("size")
    arr = df.groupby("race_id" if "race_id" in df else ["race_date", "meeting_number", "race_number"]).arrival.first()
    arrivals = {}
    for (d, m, r), a in arr.items():
        arrivals[f"R{m}C{r}_{d}"] = [int(x) for x in ast.literal_eval(a)]
    out["finish_position"] = np.nan
    for rid, a in arrivals.items():
        for pos, hn in enumerate(a, 1):
            out.loc[(out.race_id == rid) & (out.horse_number == hn), "finish_position"] = pos
    return out.reset_index(drop=True), arrivals


def insample_quintes(h: pd.DataFrame) -> list[str]:
    s = pd.read_csv(ROOT / "quinte_supports_99.csv", encoding="utf-8-sig")
    ids = ("R" + s.meeting_number.astype(str) + "C" + s.race_number.astype(str) + "_" + s.race_date).tolist()
    ok = []
    for rid in ids:
        g = h[h.race_id == rid]
        pos = pd.to_numeric(g.finish_position, errors="coerce")
        if len(g) and sorted(pos[pos.between(1, 5)].astype(int)) == [1, 2, 3, 4, 5]:
            ok.append(rid)
    return ok


def arrival_of(race: pd.DataFrame) -> list[int]:
    p = pd.to_numeric(race.finish_position, errors="coerce")
    r = race.assign(p=p)[p.between(1, 5)].sort_values("p")
    return r.horse_number.astype(int).tolist()


def shortlist_metrics(top7: list[int], order: list[int], arrival: list[int]) -> dict:
    s = set(top7[:7])
    hits = len(s & set(arrival[:5]))
    w = arrival[0]
    rank_w = order.index(w) + 1 if w in order else 99
    return {
        "hits7": hits,
        "trio7": set(arrival[:3]) <= s,
        "quarte7": set(arrival[:4]) <= s,
        "quinte7": set(arrival[:5]) <= s,
        "win_top1": rank_w <= 1, "win_top3": rank_w <= 3, "win_top7": rank_w <= 7,
        "hits5_in5": len(set(order[:5]) & set(arrival[:5])),
    }


def summarise(rows: pd.DataFrame, prefix: str = "") -> dict:
    n = len(rows)
    return {
        "races": n,
        "hits7_mean": round(rows.hits7.mean(), 3),
        "trio7": int(rows.trio7.sum()), "quarte7": int(rows.quarte7.sum()), "quinte7": int(rows.quinte7.sum()),
        "win_top1": int(rows.win_top1.sum()), "win_top3": int(rows.win_top3.sum()),
        "hits5_in5_mean": round(rows.hits5_in5.mean(), 3),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    boot = d[idx].mean(1)
    return {"diff": round(d.mean(), 4), "ci95": [round(np.quantile(boot, .025), 4), round(np.quantile(boot, .975), 4)],
            "p_gt0": round((boot > 0).mean(), 3)}
