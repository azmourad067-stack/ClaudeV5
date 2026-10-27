"""Évaluation finale V5 vs V4.5.

A) 58 Quintés walk-forward (mêmes courses que les backtests V4.x) : V5 réentraîné chaque
   semaine sur les seules courses antérieures. Avantage structurel à V4.x : ses recettes
   ont été choisies sur ces 58 courses.
B) 41 Quintés forward (12/09 -> 22/10) : jamais vus par aucune version. V5 entraîné sur
   l'historique < 12/09. Cote T-20 (conditions réelles) et cote DB.
C) 2 200 courses forward toutes disciplines (gagnant + placés PMU) : calibration P(gagner),
   comparaison au marché T-20, test de valeur au rapport PMU réel.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import pandas as pd
from common import ROOT, load_history, load_forward, paired_bootstrap
from horseprono_v5 import build_features, fit_exploded_logit, softmax_by_race, plackett_luce_topk
from v5_experiments import features_cache, GROUPS

FEATS = GROUPS["market"] + GROUPS["internal"] + GROUPS["music"]


def fit_heads(h, f, cutoff):
    m = (h.race_date < cutoff).to_numpy()
    X = f.loc[m, FEATS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1
    Z = (X - mu) / sd
    g = h.race_id.astype(str).to_numpy()[m]
    pos = pd.to_numeric(h.finish_position, errors="coerce").fillna(99).astype(int).to_numpy()[m]
    b_win = fit_exploded_logit(Z, g, pos, k=1)
    b_top5 = fit_exploded_logit(Z, g, pos, k=5)
    return {"mu": mu, "sd": sd, "win": b_win, "top5": b_top5}


def predict(model, feats: pd.DataFrame, race_ids: np.ndarray, numbers: np.ndarray):
    Z = (feats[FEATS].to_numpy(float) - model["mu"]) / model["sd"]
    pw = softmax_by_race(Z @ model["win"], race_ids)
    ps = softmax_by_race(Z @ model["top5"], race_ids)
    out = pd.DataFrame({"race_id": race_ids, "horse_number": numbers, "p_win": pw, "p_strength5": ps})
    out["p_top5"] = 0.0
    out["p_top3"] = 0.0
    for rid, idx in out.groupby("race_id").groups.items():
        s = out.loc[idx, "p_strength5"].to_numpy()
        out.loc[idx, "p_top5"] = plackett_luce_topk(s, k=min(5, len(s)), n_sim=20000)
        out.loc[idx, "p_top3"] = plackett_luce_topk(s, k=min(3, len(s)), n_sim=20000, seed=3)
    return out


def quinte_rows(pred: pd.DataFrame, arrivals: dict[str, list[int]]):
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        arr = arrivals[rid]
        g = g.sort_values(["p_top5", "p_win"], ascending=False)
        top7 = g.horse_number.astype(int).tolist()[:7]
        order = g.head(7).sort_values("p_win", ascending=False).horse_number.astype(int).tolist() + \
            g.horse_number.astype(int).tolist()[7:]
        s = set(top7)
        w = arr[0]
        rw = order.index(w) + 1 if w in order else 99
        y5 = g.horse_number.isin(arr).to_numpy()
        p5 = np.clip(g.p_top5.to_numpy(), 1e-4, 1 - 1e-4)
        pwin = g.set_index("horse_number").p_win
        rows.append({"race_id": rid, "hits7": len(s & set(arr)), "trio7": set(arr[:3]) <= s,
                     "quarte7": set(arr[:4]) <= s, "quinte7": set(arr) <= s,
                     "win_top1": rw <= 1, "win_top3": rw <= 3, "win_top7": rw <= 7,
                     "hits5_in5": len(set(order[:5]) & set(arr)),
                     "top5_ll": float(-np.mean(y5 * np.log(p5) + (1 - y5) * np.log(1 - p5))),
                     "win_ll": float(-np.log(max(pwin.get(w, 1e-6), 1e-6))),
                     "v5_top7": "-".join(map(str, order[:7])), "arrival": "-".join(map(str, arr))})
    return pd.DataFrame(rows)


def summarise(df):
    return {"races": len(df), "hits7_mean": round(df.hits7.mean(), 3), "trio7": int(df.trio7.sum()),
            "quarte7": int(df.quarte7.sum()), "quinte7": int(df.quinte7.sum()), "win_top1": int(df.win_top1.sum()),
            "win_top3": int(df.win_top3.sum()), "hits5_in5_mean": round(df.hits5_in5.mean(), 3)}


def baseline_top5_ll(hz: pd.DataFrame):
    out = {}
    for rid, g in hz.groupby("race_id"):
        p = np.clip(g.top5_prob.to_numpy(), 1e-4, 1 - 1e-4)
        y = g.y_top5.to_numpy()
        out[rid] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return out


if __name__ == "__main__":
    t0 = time.time()
    h, f = features_cache()
    h["race_date"] = pd.to_datetime(h.race_date)
    report = {}

    # ---------------- A) 58 Quintés walk-forward -----------------
    b42 = pd.read_csv(ROOT / "v42_backtest_58.csv")
    b44 = pd.read_csv(ROOT / "v44_backtest_58.csv")
    ids = b42.race_id.tolist()
    rows = []
    hq = h[h.race_id.isin(ids)].copy()
    weeks = sorted(set(pd.to_datetime(b42.race_date).dt.to_period("W-SUN").dt.start_time))
    for wk in weeks:
        model = fit_heads(h, f, wk)
        sub = hq[(hq.race_date >= wk) & (hq.race_date < wk + pd.Timedelta(days=7))]
        fs = f.loc[sub.index]
        pred = predict(model, fs, sub.race_id.astype(str).to_numpy(), sub.horse_number.astype(int).to_numpy())
        arrivals = {rid: g.assign(p=pd.to_numeric(g.finish_position, errors="coerce")).query("p<=5").sort_values("p")
                    .horse_number.astype(int).tolist() for rid, g in sub.groupby("race_id")}
        rows.append(quinte_rows(pred, arrivals))
        print("week", wk.date(), "done", round(time.time() - t0), flush=True)
    A = pd.concat(rows).set_index("race_id").loc[ids].reset_index()
    A["discipline"] = b42.discipline.values
    A.to_csv(ROOT / "research" / "v5_walkforward58.csv", index=False)
    base = pd.DataFrame({"race_id": ids, "hits7": b42.v42_hits, "trio7": b42.v42_trio, "quarte7": b42.v42_quarte,
                         "quinte7": b42.v42_quinte, "win_rank": b44.v44_winner_rank})
    base["win_top1"] = base.win_rank <= 1
    base["win_top3"] = base.win_rank <= 3
    report["A_walkforward58"] = {
        "V4.5 (backtest publié, recettes réglées sur ces courses)": {
            "races": 58, "hits7_mean": round(base.hits7.mean(), 3), "trio7": int(base.trio7.sum()),
            "quarte7": int(base.quarte7.sum()), "quinte7": int(base.quinte7.sum()),
            "win_top1": int(base.win_top1.sum()), "win_top3": int(base.win_top3.sum())},
        "V5": summarise(A),
        "V5_minus_V45_hits7": paired_bootstrap(base.hits7.values, A.hits7.values),
        "V5_minus_V45_win_top3": paired_bootstrap(base.win_top3.astype(float).values, A.win_top3.astype(float).values),
    }
    flat = A.discipline.eq("PLAT").values
    report["A_walkforward58_PLAT"] = {
        "V4.5": {"races": int(flat.sum()), "hits7_mean": round(base.hits7[flat].mean(), 3),
                 "win_top3": int(base.win_top3[flat].sum()), "quarte7": int(base.quarte7[flat].sum())},
        "V5": summarise(A[flat])}

    # ---------------- B) 41 Quintés forward -----------------
    model = fit_heads(h, f, pd.Timestamp("2026-09-12"))
    json.dump({"features": FEATS, "mean": model["mu"].tolist(), "scale": model["sd"].tolist(),
               "coef_win": model["win"].tolist(), "coef_top5": model["top5"].tolist(),
               "train_end_exclusive": "2026-09-12"}, open(ROOT / "research" / "v5_model_cut0912.json", "w"))
    for src in ["snap", "db"]:
        fwd, arrivals = load_forward(src)
        ff = build_features(h, fwd.drop(columns=["finish_position"]).assign(finish_position=np.nan))
        pred = predict(model, ff, fwd.race_id.astype(str).to_numpy(), fwd.horse_number.astype(int).to_numpy())
        B = quinte_rows(pred, arrivals)
        B["discipline"] = fwd.groupby("race_id", sort=False).discipline.first().reindex(B.race_id).values
        B.to_csv(ROOT / "research" / f"v5_forward_{src}.csv", index=False)
        bl = pd.read_csv(ROOT / "research" / f"baseline_forward_{src}.csv").set_index("race_id").loc[B.race_id].reset_index()
        bhz = pd.read_csv(ROOT / "research" / f"baseline_forward_{src}_horses.csv")
        bll = baseline_top5_ll(bhz)
        blr = bl.rename(columns=lambda c: c[4:] if c.startswith("v45_") else c)
        mk = bl.rename(columns=lambda c: c[4:] if c.startswith("mkt_") else c)
        key = f"B_forward41_{src}"
        report[key] = {
            "market_top7": summarise(mk), "V4.5": summarise(blr), "V5": summarise(B),
            "top5_logloss": {"V4.5_logistic": round(np.mean([bll[r] for r in B.race_id]), 4),
                             "V5": round(B.top5_ll.mean(), 4)},
            "V5_minus_V45_hits7": paired_bootstrap(blr.hits7.values, B.hits7.values),
            "V5_minus_market_hits7": paired_bootstrap(mk.hits7.values, B.hits7.values),
            "V5_minus_V45_win_top3": paired_bootstrap(blr.win_top3.astype(float).values, B.win_top3.astype(float).values),
        }
        fl = B.discipline.eq("PLAT").values
        report[key + "_PLAT"] = {"market_top7": summarise(mk[fl]), "V4.5": summarise(blr[fl]), "V5": summarise(B[fl])}
        print(key, json.dumps(report[key]), flush=True)
    json.dump(report, open(ROOT / "research" / "v5_eval_report.json", "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    print("sec", round(time.time() - t0))
