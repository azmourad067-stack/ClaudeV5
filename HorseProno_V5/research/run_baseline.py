"""Rejoue le pipeline V4.5 tel qu'en production (V4.2 Top7 -> V4.3 shadow -> V4.4 -> V4.5)."""
from __future__ import annotations
import json, sys, time
import numpy as np
import pandas as pd
from common import ROOT, load_history, load_forward, insample_quintes, arrival_of, shortlist_metrics, summarise

from quinte_v2 import load_artifact
from quinte_v3 import load_ranker
from quinte_v43 import load_outsider_artifact
from quinte_v44 import rank_quinte_v44
from quinte_v45 import apply_v45_market_dynamics

ART = load_artifact(ROOT / "quinte_v2_artifact.json")
RANKER = load_ranker(ROOT / "quinte_v2_ranker.txt")
OUT = load_outsider_artifact(ROOT / "v43_outsider_artifact.json")


def run_one(history, race):
    hist = history[history.race_date < race.race_date.iloc[0]]
    res = rank_quinte_v44(hist, race, ART, RANKER, OUT)
    res = apply_v45_market_dynamics(res, snapshot_history=None, minutes_to_start=None)
    return res


def evaluate(history, races: dict[str, pd.DataFrame], label: str):
    rows, horses = [], []
    for rid, race in races.items():
        arrival = arrival_of(race)
        race_in = race.drop(columns=["finish_position"]).copy()
        race_in["finish_position"] = np.nan
        res = run_one(history, race_in)
        order = res.horse_number.astype(int).tolist()
        top7 = res.loc[res.selected_top7.astype(bool), "horse_number"].astype(int).tolist()
        mkt = race.assign(o=pd.to_numeric(race.odds, errors="coerce").fillna(999)).sort_values(["o", "horse_number"]).horse_number.astype(int).tolist()
        m = shortlist_metrics(top7, order, arrival)
        mm = shortlist_metrics(mkt[:7], mkt, arrival)
        rows.append({"race_id": rid, "discipline": race.discipline.iloc[0], "field": len(race),
                     **{f"v45_{k}": v for k, v in m.items()}, **{f"mkt_{k}": v for k, v in mm.items()},
                     "v45_top7": "-".join(map(str, top7)), "arrival": "-".join(map(str, arrival))})
        fin = dict(zip(arrival, range(1, 6)))
        for r in res.itertuples():
            horses.append({"race_id": rid, "horse_number": int(r.horse_number), "top5_prob": r.top5_probability,
                           "market_prob": r.market_probability, "y_top5": int(int(r.horse_number) in fin),
                           "y_win": int(fin.get(int(r.horse_number)) == 1)})
    df = pd.DataFrame(rows)
    return df, pd.DataFrame(horses)


def to_summary(df, p):
    s = df[[c for c in df.columns if c.startswith(p)]].rename(columns=lambda c: c[len(p):])
    return summarise(s)


if __name__ == "__main__":
    t = time.time()
    h = load_history()
    results = {}
    # 1) Forward : 41 Quintés jamais vus (cote T-20 et cote DB)
    for src in ["snap", "db"]:
        fwd, arrivals = load_forward(src)
        races = {rid: g.reset_index(drop=True) for rid, g in fwd.groupby("race_id", sort=False)}
        df, hz = evaluate(h, races, f"forward_{src}")
        df.to_csv(ROOT / "research" / f"baseline_forward_{src}.csv", index=False)
        hz.to_csv(ROOT / "research" / f"baseline_forward_{src}_horses.csv", index=False)
        results[f"forward_{src}"] = {"V4.5": to_summary(df, "v45_"), "market": to_summary(df, "mkt_")}
        flat = df[df.discipline == "PLAT"]
        results[f"forward_{src}_PLAT"] = {"V4.5": to_summary(flat, "v45_"), "market": to_summary(flat, "mkt_")}
    # 2) In-sample : 98 Quintés (artefact entraîné dessus -> optimiste)
    ids = insample_quintes(h)
    races = {rid: h[h.race_id == rid].reset_index(drop=True) for rid in ids}
    races = {k: v[~v.is_non_runner.astype(bool)].reset_index(drop=True) for k, v in races.items()}
    df, hz = evaluate(h, races, "insample")
    df.to_csv(ROOT / "research" / "baseline_insample.csv", index=False)
    hz.to_csv(ROOT / "research" / "baseline_insample_horses.csv", index=False)
    results["insample_98"] = {"V4.5": to_summary(df, "v45_"), "market": to_summary(df, "mkt_")}
    print(json.dumps(results, indent=1))
    print("sec", round(time.time() - t))
