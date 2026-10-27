"""Côté V5 et marché du test grands champs (cf. bigfield_compare.py)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common import ROOT, paired_bootstrap, shortlist_metrics, arrival_of
from v5_experiments import features_cache
from v5_eval import fit_heads, predict

if __name__ == "__main__":
    h, f = features_cache()
    h["race_date"] = pd.to_datetime(h.race_date)
    v45 = pd.read_csv(ROOT / "research" / "bigfield_v45.csv")
    ids = v45.race_id.tolist()
    model = fit_heads(h, f, pd.Timestamp("2026-08-01"))
    sub = h[h.race_id.isin(ids)]
    pred = predict(model, f.loc[sub.index], sub.race_id.astype(str).to_numpy(), sub.horse_number.astype(int).to_numpy())
    rows, mrows = [], []
    for rid, g in pred.groupby("race_id"):
        race = sub[sub.race_id == rid]
        arr = arrival_of(race)
        g = g.sort_values(["p_top5", "p_win"], ascending=False)
        top7 = g.horse_number.astype(int).tolist()[:7]
        order = g.head(7).sort_values("p_win", ascending=False).horse_number.astype(int).tolist() + g.horse_number.astype(int).tolist()[7:]
        rows.append({"race_id": rid, **shortlist_metrics(top7, order, arr)})
        mk = race.assign(o=pd.to_numeric(race.odds, errors="coerce").fillna(999)).sort_values(["o", "horse_number"]).horse_number.astype(int).tolist()
        mrows.append({"race_id": rid, **shortlist_metrics(mk[:7], mk, arr)})
    V5 = pd.DataFrame(rows).set_index("race_id").loc[ids].reset_index()
    MK = pd.DataFrame(mrows).set_index("race_id").loc[ids].reset_index()
    disc = h.groupby("race_id").discipline.first().loc[ids].values
    out = {}
    for name, mask in [("all", np.ones(len(ids), bool)), ("PLAT", disc == "PLAT"),
                       ("TROT", np.isin(disc, ["ATTELE_VOLTE", "ATTELE_AUTOSTART", "TROT_MONTE"]))]:
        a, b, c = MK[mask], v45[mask], V5[mask]
        s = lambda d: {"races": int(len(d)), "hits7": round(d.hits7.mean(), 3), "trio7": round(d.trio7.mean(), 3),
                       "quarte7": round(d.quarte7.mean(), 3), "quinte7": round(d.quinte7.mean(), 3),
                       "win_top1": round(d.win_top1.mean(), 3), "win_top3": round(d.win_top3.mean(), 3)}
        out[name] = {"market": s(a), "V4.5": s(b), "V5": s(c),
                     "V45_minus_market_hits7": paired_bootstrap(a.hits7.values, b.hits7.values),
                     "V5_minus_market_hits7": paired_bootstrap(a.hits7.values, c.hits7.values),
                     "V5_minus_V45_hits7": paired_bootstrap(b.hits7.values, c.hits7.values),
                     "V5_minus_V45_quarte7": paired_bootstrap(b.quarte7.astype(float).values, c.quarte7.astype(float).values),
                     "V5_minus_V45_win_top3": paired_bootstrap(b.win_top3.astype(float).values, c.win_top3.astype(float).values)}
    json.dump(out, open(ROOT / "research" / "bigfield_report.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
