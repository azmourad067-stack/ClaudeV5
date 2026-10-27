"""Validation (01/08 -> 11/09, toutes courses) des métriques Top5/shortlist pour quelques variantes."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common import ROOT, paired_bootstrap
from v5_experiments import features_cache, run, GROUPS
from horseprono_v5 import FEATURES_V5, plackett_luce_topk


def race_metrics(hv, p, n_sim=4000):
    rows = []
    d = hv.assign(p=p, pos=pd.to_numeric(hv.finish_position, errors="coerce"))
    for rid, g in d.groupby("race_id", sort=False):
        if len(g) < 8:
            continue
        pw = g.p.to_numpy(); pw = pw / pw.sum()
        p5 = plackett_luce_topk(pw / pw.sum(), k=5, n_sim=n_sim)
        y5 = g.pos.between(1, 5).to_numpy()
        order = np.lexsort((-pw, -p5))
        top7 = set(g.horse_number.to_numpy()[order[:7]])
        arr = g.sort_values("pos").horse_number.to_numpy()[:5]
        winner_p = pw[g.pos.to_numpy() == 1]
        rows.append({
            "race_id": rid, "field": len(g), "disc": g.discipline.iloc[0],
            "win_ll": -np.log(max(winner_p[0], 1e-9)) if len(winner_p) else np.nan,
            "top5_ll": -np.mean(y5 * np.log(np.clip(p5, 1e-4, 1)) + (1 - y5) * np.log(np.clip(1 - p5, 1e-4, 1))),
            "hits7": len(top7 & set(arr)),
            "quarte7": set(arr[:4]) <= top7,
            "win_top1": int(np.argmax(pw) == np.flatnonzero(g.pos.to_numpy() == 1)[0]) if len(winner_p) else 0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    h, f = features_cache()
    d = h.race_date
    train = (d < "2026-08-01").to_numpy()
    valid = ((d >= "2026-08-01") & (d <= "2026-09-11")).to_numpy()
    hv = h[valid].reset_index(drop=True)
    mk = np.exp(f.loc[valid, "log_mkt"].to_numpy())
    variants = {"market_raw": mk}
    specs = {
        "market_fit_k1": (GROUPS["market"], 1, None),
        "market_fit_k5": (GROUPS["market"], 5, None),
        "market+music_k1": (GROUPS["market"] + GROUPS["music"], 1, None),
        "market+music_k5": (GROUPS["market"] + GROUPS["music"], 5, None),
        "mkt+int+music_k1": (GROUPS["market"] + GROUPS["internal"] + GROUPS["music"], 1, None),
        "mkt+int+music_k5w": (GROUPS["market"] + GROUPS["internal"] + GROUPS["music"], 5, [1, .6, .4, .3, .2]),
        "mkt+int+music_k5": (GROUPS["market"] + GROUPS["internal"] + GROUPS["music"], 5, None),
        "all_k5": (FEATURES_V5, 5, None),
    }
    for name, (fl, k, sw) in specs.items():
        variants[name] = run(fl, h, f, train, valid, k=k, stage_weights=sw)[3]
    out, per = {}, {}
    for name, p in variants.items():
        m = race_metrics(hv, p)
        per[name] = m
        big = m[m.field >= 12]
        out[name] = {"races>=8": len(m), "win_ll": round(m.win_ll.mean(), 4), "top5_ll": round(m.top5_ll.mean(), 4),
                     "hits7": round(m.hits7.mean(), 3), "quarte7": round(m.quarte7.mean(), 3),
                     "win_top1": round(m.win_top1.mean(), 3),
                     "big12_races": len(big), "big12_hits7": round(big.hits7.mean(), 3), "big12_top5_ll": round(big.top5_ll.mean(), 4)}
        print(name, out[name], flush=True)
    base = per["market_raw"]
    for name in variants:
        if name == "market_raw":
            continue
        out[name]["vs_market_hits7"] = paired_bootstrap(base.hits7, per[name].hits7)
        out[name]["vs_market_top5_ll(neg=better)"] = paired_bootstrap(base.top5_ll, per[name].top5_ll)
        out[name]["vs_market_win_ll(neg=better)"] = paired_bootstrap(base.win_ll.fillna(0), per[name].win_ll.fillna(0))
        print(name, {k: v for k, v in out[name].items() if k.startswith("vs")}, flush=True)
    json.dump(out, open(ROOT / "research" / "v5_dev_top5.json", "w"), indent=1, default=str)
