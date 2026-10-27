"""C) Forward toutes courses (12/09 -> 23/10/2026) : gagnant + placés issus des rapports PMU.
Cote T-20 (forward_model_predictions) = information réellement disponible avant la course.
Paris réglés au rapport PMU définitif (dividende pour 1 €), pas à la cote T-20."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common import ROOT, paired_bootstrap
from horseprono_v5 import build_features, softmax_by_race, plackett_luce_topk
from v5_experiments import features_cache

FEATS = json.load(open(ROOT / "research" / "v5_model_cut0912.json"))
MODEL = {k: np.asarray(v) for k, v in FEATS.items() if k in ("mean", "scale", "coef_win", "coef_top5")}


def load_all():
    df = pd.read_csv(ROOT / "data" / "forward_allraces_raw.csv", keep_default_na=False, dtype=str)
    df = df[df.is_non_runner.astype(str) == "0"].copy()
    num = lambda c: pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    out = pd.DataFrame({
        "race_id": "R" + df.meeting_number + "C" + df.race_number + "_" + df.race_date,
        "race_date": pd.to_datetime(df.race_date), "discipline": df.discipline, "hippodrome": df.hippodrome,
        "distance": num("distance"), "horse_number": num("horse_number").astype(int), "horse_name": df.horse_name,
        "jockey": df.jockey, "trainer": df.trainer, "odds": num("odds_snap"), "draw": num("draw"),
        "weight": num("weight"), "age": num("age"), "sex": df.sex, "recent_form": df.recent_form,
        "win": df.win, "win_div": num("win_div"), "place": df.place, "is_non_runner": False,
    })
    out["field_size"] = out.groupby("race_id").horse_number.transform("size")
    out["is_winner"] = [str(n) in str(w).split(",") for n, w in zip(out.horse_number, out.win)]
    pl = {}
    for n, p in zip(out.horse_number, out.place):
        pass
    out["placed"] = [any(tok.split(":")[0] == str(n) for tok in str(p).split(";") if tok) for n, p in zip(out.horse_number, out.place)]
    out["place_div"] = [next((float(tok.split(":")[1]) for tok in str(p).split(";") if tok and tok.split(":")[0] == str(n)), np.nan)
                        for n, p in zip(out.horse_number, out.place)]
    ok = out.groupby("race_id").agg(full_odds=("odds", lambda s: s.notna().all() and (s > 1).all()),
                                    nwin=("is_winner", "sum"), n=("horse_number", "size"))
    keep = ok[(ok.full_odds) & (ok.nwin == 1) & (ok.n >= 4)].index
    return out[out.race_id.isin(keep)].reset_index(drop=True)


if __name__ == "__main__":
    h, _ = features_cache()
    h["race_date"] = pd.to_datetime(h.race_date)
    d = load_all()
    feats = build_features(h, d.assign(finish_position=np.nan))
    Z = (feats[FEATS["features"]].to_numpy(float) - MODEL["mean"]) / MODEL["scale"]
    rid = d.race_id.to_numpy()
    d["p_model"] = softmax_by_race(Z @ MODEL["coef_win"], rid)
    s5 = softmax_by_race(Z @ MODEL["coef_top5"], rid)
    inv = 1 / d.odds
    d["p_mkt"] = inv / inv.groupby(d.race_id).transform("sum")
    d["overround"] = inv.groupby(d.race_id).transform("sum")
    # P(placé) : top3 si >=8 partants, top2 si 4..7 (règle PMU simple placé)
    d["p_place_model"] = 0.0
    d["p_place_mkt"] = 0.0
    for r, idx in d.groupby("race_id").groups.items():
        k = 3 if len(idx) >= 8 else 2
        d.loc[idx, "p_place_model"] = plackett_luce_topk(s5[idx], k=k, n_sim=8000)
        d.loc[idx, "p_place_mkt"] = plackett_luce_topk(d.loc[idx, "p_mkt"].to_numpy(), k=k, n_sim=8000)
    races = d.race_id.nunique()
    w = d[d.is_winner]
    ll_model = -np.log(w.p_model.clip(1e-6))
    ll_mkt = -np.log(w.p_mkt.clip(1e-6))
    rep = {"races": int(races), "runners": len(d), "period": [str(d.race_date.min().date()), str(d.race_date.max().date())],
           "mean_overround_T20": round(d.groupby("race_id").overround.first().mean(), 3),
           "win_logloss": {"market_T20": round(ll_mkt.mean(), 4), "V5": round(ll_model.mean(), 4),
                           "V5_minus_market": paired_bootstrap(ll_mkt.values, ll_model.values)},
           "fav_wins": {"market": round(d.loc[d.groupby("race_id").p_mkt.idxmax()].is_winner.mean(), 4),
                        "V5": round(d.loc[d.groupby("race_id").p_model.idxmax()].is_winner.mean(), 4)}}
    yp = d.placed.astype(float)
    pl = lambda p: -(yp * np.log(p.clip(1e-4, 1 - 1e-4)) + (1 - yp) * np.log((1 - p).clip(1e-4, 1 - 1e-4)))
    pr_m = pl(d.p_place_model).groupby(d.race_id).mean()
    pr_k = pl(d.p_place_mkt).groupby(d.race_id).mean()
    rep["place_logloss"] = {"market_PL": round(pr_k.mean(), 4), "V5_top5head_PL": round(pr_m.mean(), 4),
                            "V5_minus_market": paired_bootstrap(pr_k.values, pr_m.loc[pr_k.index].values)}
    # Calibration
    d["bin"] = pd.cut(d.p_model, [0, .02, .05, .1, .2, .3, .5, 1])
    cal = d.groupby("bin", observed=True).agg(n=("is_winner", "size"), pred=("p_model", "mean"),
                                              obs=("is_winner", "mean"), mkt=("p_mkt", "mean"))
    rep["calibration_V5_win"] = cal.round(4).reset_index().astype(str).to_dict("records")
    # Valeur : mise 1 € quand p_model * cote_T20 > seuil, réglée au rapport PMU réel
    d["ret"] = np.where(d.is_winner, d.win_div.fillna(0), 0.0)
    val = {"all_runners": {"bets": len(d), "roi": round(d.ret.sum() / len(d) - 1, 4)}}
    for thr in [1.0, 1.1, 1.2, 1.3, 1.5]:
        b = d[d.p_model * d.odds > thr]
        rng = np.random.default_rng(0)
        boots = [b.ret.sample(len(b), replace=True, random_state=i).mean() - 1 for i in range(2000)] if len(b) else [0]
        val[f"edge>{thr}"] = {"bets": len(b), "winners": int(b.is_winner.sum()), "roi": round(b.ret.mean() - 1, 4) if len(b) else None,
                              "roi_ci95": [round(np.quantile(boots, .025), 3), round(np.quantile(boots, .975), 3)]}
    rep["value_win_bets"] = val
    # Par discipline
    rep["by_discipline_win_ll"] = {k: {"races": int(g.race_id.nunique()), "market": round(-np.log(g.p_mkt).mean(), 4),
                                       "V5": round(-np.log(g.p_model).mean(), 4)} for k, g in w.groupby("discipline")}
    d.to_csv(ROOT / "research" / "v5_forward_allraces_preds.csv", index=False)
    json.dump(rep, open(ROOT / "research" / "v5_eval_allraces.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
