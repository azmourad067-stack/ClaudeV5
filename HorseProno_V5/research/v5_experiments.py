"""Développement V5 : sélection de features / hyperparamètres sur un bloc de validation
temporel (toutes courses), sans jamais toucher aux 41 Quintés forward."""
from __future__ import annotations
import json, sys, time
import numpy as np
import pandas as pd
from common import ROOT, load_history
from horseprono_v5 import build_features, fit_exploded_logit, softmax_by_race, FEATURES_V5

GROUPS = {
    "market": ["log_mkt", "log_mkt_trot", "odds_missing"],
    "internal": ["horse_name_top5", "horse_name_win", "jockey_top5", "jockey_win", "trainer_top5", "trainer_win",
                 "horse_name_starts", "jockey_starts", "trainer_starts", "form", "consistency"],
    "music": ["mu_n", "mu_last", "mu_mean3", "mu_mean6", "mu_win6", "mu_top3_6", "mu_fail6", "mu_fail6_trot",
              "mu_same_disc6", "mu_same_disc6_obst", "mu_layoff", "mu_missing"],
    "physical": ["weight_galop", "draw_plat", "draw_plat_sq", "draw_autostart", "age_young", "age_old", "sex_FEMELLES"],
}


def features_cache():
    p = ROOT / "research" / "feat_history.pkl"
    h = load_history()
    h = h[~h.is_non_runner.astype(bool)].reset_index(drop=True)
    if p.exists():
        f = pd.read_pickle(p)
    else:
        t = time.time()
        f = build_features(h, h)
        f.to_pickle(p)
        print("features", round(time.time() - t), "s", file=sys.stderr)
    return h, f


def standardise(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-9] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def evaluate_probs(df: pd.DataFrame, p: np.ndarray) -> dict:
    d = df.assign(p=p)
    pos = pd.to_numeric(d.finish_position, errors="coerce")
    d["win"] = (pos == 1).astype(int)
    d["top5"] = pos.between(1, 5).astype(int)
    w = d[d.win == 1]
    win_ll = -np.log(np.clip(w.p.to_numpy(), 1e-9, 1)).mean()
    return {"races": int(d.race_id.nunique()), "win_logloss": round(float(win_ll), 4)}


def run(feature_list, h, f, train_mask, test_mask, k=5, l2=1.0, stage_weights=None):
    X = f[feature_list].to_numpy(float)
    g = h.race_id.astype(str).to_numpy()
    pos = pd.to_numeric(h.finish_position, errors="coerce").fillna(99).astype(int).to_numpy()
    Xtr, Xte, mu, sd = standardise(X[train_mask], X[test_mask])
    beta = fit_exploded_logit(Xtr, g[train_mask], pos[train_mask], k=k, l2=l2, stage_weights=stage_weights)
    p = softmax_by_race(Xte @ beta, g[test_mask])
    return beta, mu, sd, p


if __name__ == "__main__":
    h, f = features_cache()
    d = h.race_date
    train = (d < "2026-08-01").to_numpy()
    valid = ((d >= "2026-08-01") & (d <= "2026-09-11")).to_numpy()
    hv = h[valid].reset_index(drop=True)
    print("train races", h[train].race_id.nunique(), "valid races", hv.race_id.nunique())
    res = {}
    # marché brut (proba implicite normalisée) pour référence
    mk = np.exp(f.loc[valid, "log_mkt"].to_numpy())
    mk = mk / pd.Series(mk).groupby(hv.race_id.to_numpy()).transform("sum").to_numpy()
    res["market_raw"] = evaluate_probs(hv, mk)
    configs = {
        "market_fit_k1": (GROUPS["market"], 1),
        "market+internal_k1": (GROUPS["market"] + GROUPS["internal"], 1),
        "market+music_k1": (GROUPS["market"] + GROUPS["music"], 1),
        "market+internal+music_k1": (GROUPS["market"] + GROUPS["internal"] + GROUPS["music"], 1),
        "all_k1": (FEATURES_V5, 1),
        "all_k3": (FEATURES_V5, 3),
        "all_k5": (FEATURES_V5, 5),
    }
    betas = {}
    for name, (fl, k) in configs.items():
        beta, mu, sd, p = run(fl, h, f, train, valid, k=k)
        res[name] = evaluate_probs(hv, p)
        betas[name] = dict(zip(fl, np.round(beta, 3)))
        print(name, res[name], flush=True)
    for sw in ([1, .8, .6, .5, .4], [1, .6, .4, .3, .2]):
        beta, mu, sd, p = run(FEATURES_V5, h, f, train, valid, k=5, stage_weights=sw)
        res[f"all_k5_w{sw}"] = evaluate_probs(hv, p)
        print(f"all_k5_w{sw}", res[f"all_k5_w{sw}"], flush=True)
    for l2 in (10.0, 100.0):
        beta, mu, sd, p = run(FEATURES_V5, h, f, train, valid, k=5, l2=l2)
        res[f"all_k5_l2_{l2}"] = evaluate_probs(hv, p)
        print(f"all_k5_l2_{l2}", res[f"all_k5_l2_{l2}"], flush=True)
    json.dump({"results": res, "betas": {k: {a: float(b) for a, b in v.items()} for k, v in betas.items()}},
              open(ROOT / "research" / "v5_dev_results.json", "w"), indent=1)
