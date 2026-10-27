"""Test à forte puissance : V4.5 vs V5 vs marché sur les courses à >=12 partants hors Quinté,
du 01/08 au 11/09/2026. Ni V4.x (entraîné/réglé sur les seuls Quintés) ni V5 (entraîné < 01/08)
n'ont vu ces courses. Même historique, même cote."""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from common import ROOT, load_history, insample_quintes, arrival_of, shortlist_metrics, paired_bootstrap
from fast_hist import FastHistory

H = None
FH = None


def _init():
    global H, FH
    H = load_history()
    H = H[~H.is_non_runner.astype(bool)].reset_index(drop=True)
    FH = FastHistory(H)


def run_v45(rid):
    from run_baseline import run_one
    race = H[H.race_id == rid].reset_index(drop=True)
    arr = arrival_of(race)
    q = race.drop(columns=["finish_position"]).assign(finish_position=np.nan)
    res = run_one(FH.subset(race), q)
    order = res.horse_number.astype(int).tolist()
    top7 = res.loc[res.selected_top7.astype(bool), "horse_number"].astype(int).tolist()
    m = shortlist_metrics(top7, order, arr)
    return {"race_id": rid, **m}


def check():
    _init()
    from run_baseline import run_one
    rid = H[H.race_date == "2026-08-20"].race_id.iloc[0]
    race = H[H.race_id == rid].reset_index(drop=True).assign(finish_position=np.nan)
    a = run_one(H, race)[["horse_number", "top5_probability", "ranker_score"]]
    b = run_one(FH.subset(race), race)[["horse_number", "top5_probability", "ranker_score"]]
    print("identical:", np.allclose(a.to_numpy(float), b.to_numpy(float)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(); sys.exit()
    _init()
    q = set(insample_quintes(H))
    g = H.groupby("race_id").agg(d=("race_date", "first"), n=("horse_number", "size"))
    ids = g[(g.d >= "2026-08-01") & (g.d <= "2026-09-11") & (g.n >= 12) & (~g.index.isin(q))].index.tolist()
    print("races", len(ids), flush=True)
    t = time.time()
    with Pool(2, initializer=_init) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(run_v45, ids, chunksize=4)):
            rows.append(r)
            if i % 50 == 0:
                print(i, round(time.time() - t), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "research" / "bigfield_v45.csv", index=False)
    print("done", round(time.time() - t))
