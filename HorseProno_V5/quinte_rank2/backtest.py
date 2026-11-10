"""Backtest walk-forward : 2e du modèle HorseProno V5 au simple gagnant / simple placé, quintés.

Protocole anti-fuite
- Modèle : réentraîné chaque lundi (train_v5 : mêmes features FEATURES_PROD, même
  logit exploded, l2=1, température 1,10) sur les SEULES courses de date < ce lundi.
  Le quinté prédit (date ≥ lundi) n'est jamais dans l'apprentissage. v5_artifact.json
  du dépôt (appris jusqu'au 16/09/2026) n'est PAS utilisé.
- Inférence : V5Model.predict_race du dépôt, historique restreint aux courses de date
  strictement antérieure au quinté (stats cheval/jockey/entraîneur, forme).
- Cotes : cote de référence PMU publiée avant le départ (contrôlé dans build_dataset).
- Musique : champ PMU pré-course (contrôle d'antériorité imprimé ci-dessous).
- Règlement : rapports définitifs officiels ; course sans rapport exclue et comptée.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horseprono_v5 import FEATURES_PROD, V5Model, build_features, fit_exploded_logit, parse_music  # noqa: E402

WIN_TEMPERATURE = 1.10  # train_v5.py


def load_history(path):
    h = pd.read_csv(path, low_memory=False)
    h["race_date"] = pd.to_datetime(h["race_date"]).dt.normalize()
    return h[~h["is_non_runner"].astype(bool)].reset_index(drop=True)


def fit_artifact(h, feats, cutoff):
    m = (h.race_date < cutoff).to_numpy()
    X = feats.loc[m, FEATURES_PROD].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    g = h.race_id.astype(str).to_numpy()[m]
    pos = pd.to_numeric(h.finish_position, errors="coerce").fillna(99).astype(int).to_numpy()[m]
    return {"model_name": "HorseProno_V5_walkforward", "features": FEATURES_PROD,
            "mean": mu.tolist(), "scale": sd.tolist(),
            "coef_win": fit_exploded_logit(Z, g, pos, k=1).tolist(),
            "coef_top5": fit_exploded_logit(Z, g, pos, k=5).tolist(),
            "win_temperature": WIN_TEMPERATURE, "training_races": int(len(set(g))),
            "training_end": str(h.race_date[m].max().date()), "cutoff": str(cutoff.date())}


def market_order(race):
    o = pd.to_numeric(race.odds, errors="coerce")
    r = race.assign(_o=o)[o > 1].sort_values(["_o", "horse_number"])
    return r.horse_number.astype(int).tolist()


def predictions(h, quintes, out_csv):
    feats = build_features(h, h)  # stats strictement antérieures à chaque course (merge_asof strict)
    done = pd.read_csv(out_csv) if Path(out_csv).exists() else pd.DataFrame()
    have = set(done.race_id) if len(done) else set()
    rows, models = [], {}
    for q in quintes.itertuples():
        if q.race_id in have:
            continue
        d = pd.Timestamp(q.race_date)
        cutoff = d - pd.Timedelta(days=d.weekday())  # lundi de la semaine du quinté
        if cutoff not in models:
            models[cutoff] = fit_artifact(h, feats, cutoff)
            print("modèle", cutoff.date(), models[cutoff]["training_races"], "courses", flush=True)
        art = models[cutoff]
        assert pd.Timestamp(art["training_end"]) < d
        race = h[h.race_id == q.race_id]
        if len(race) < 3:
            continue
        past = h[h.race_date < d]
        pred = V5Model(art).predict_race(past, race, n_sim=20000)
        mo = market_order(race)
        rows.append({"race_id": q.race_id, "race_date": q.race_date, "cutoff": str(cutoff.date()),
                     "train_end": art["training_end"], "n_runners": len(race),
                     "odds_missing": int(pd.to_numeric(race.odds, errors="coerce").isna().sum()),
                     "model_1": int(pred.horse_number.iloc[0]), "model_2": int(pred.horse_number.iloc[1]),
                     "market_1": mo[0] if len(mo) > 0 else None, "market_2": mo[1] if len(mo) > 1 else None,
                     "model_order": "-".join(map(str, pred.horse_number.astype(int).tolist()[:7]))})
        if len(rows) % 20 == 0:
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_csv, index=False)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(out_csv, index=False)
    return out


def boot_ci(x, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1) - 1
    return np.percentile(m, [2.5, 97.5])


def evaluate(pred, quintes):
    q = quintes.set_index("race_id")
    strategies = {"2e du modèle": "model_2", "2e de la cote": "market_2",
                  "Favori du modèle": "model_1", "Favori de la cote": "market_1"}
    res, detail = [], []
    for bet, col in [("Gagnant", "gagnant"), ("Placé", "place")]:
        ok = pred[pred.race_id.map(lambda r: isinstance(q.loc[r, col], str))].sort_values(["race_date", "race_id"])
        ok = ok.dropna(subset=list(strategies.values()))
        excluded = len(pred) - len(ok)
        half = len(ok) // 2
        for name, sc in strategies.items():
            pay = np.array([json.loads(q.loc[r, col]).get(str(int(n)), 0.0) for r, n in zip(ok.race_id, ok[sc])])
            lo, hi = boot_ci(pay)
            res.append({"Pari": bet, "Stratégie": name, "Paris": len(pay), "Exclus (sans rapport)": excluded,
                        "Réussite %": 100 * (pay > 0).mean(), "ROI %": 100 * (pay.mean() - 1),
                        "IC95 bas %": 100 * lo, "IC95 haut %": 100 * hi,
                        "ROI 1re moitié %": 100 * (pay[:half].mean() - 1),
                        "ROI 2e moitié %": 100 * (pay[half:].mean() - 1),
                        "Rapport moyen gagnant": pay[pay > 0].mean() if (pay > 0).any() else np.nan})
            detail.append(pd.Series(pay, index=ok.race_id.values, name=f"{bet}|{name}"))
        print(f"{bet} : {len(ok)} quintés, 1re moitié {ok.race_date.iloc[0]} → {ok.race_date.iloc[half-1]}, "
              f"2e moitié {ok.race_date.iloc[half]} → {ok.race_date.iloc[-1]}")
    return pd.DataFrame(res), pd.concat(detail, axis=1)


def paired_diff(detail, a, b, n=10000, seed=2):
    d = (detail[a] - detail[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def leak_checks(h, quintes):
    q = h[h.race_id.isin(quintes.race_id)]
    # 1) cotes : toutes publiées avant le départ
    used = q[q.odds.notna()]
    assert (used.odds_ref_ms < used.start_ms).all()
    # 2) musique : le 1er résultat de la musique ne doit pas être la course du jour
    first = q.recent_form.fillna("").map(lambda s: parse_music(s)["mu_last"])
    win = q.finish_full.eq(1)
    print(f"Contrôle musique : P(1er symbole = 1 | gagnant du jour) = {(first[win] == 1).mean():.2f} "
          f"vs P(1er symbole = 1 | autres) = {(first[~win] == 1).mean():.2f}")
    print(f"Cotes manquantes/non antérieures au départ : {q.odds.isna().mean():.1%} des partants de quintés ; "
          f"délai médian publication→départ {((used.start_ms - used.odds_ref_ms) / 60000).median():.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.environ.get("PMU_CACHE", "/tmp/pmu_cache") + "_tables")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--from", dest="start", default="2025-11-01")
    a = ap.parse_args()
    T, O = Path(a.tables), Path(a.out)
    O.mkdir(exist_ok=True)
    h = load_history(T / "history.csv")
    quintes = pd.read_csv(T / "quintes.csv")
    quintes = quintes[quintes.race_date >= a.start].sort_values(["race_date", "race_id"]).reset_index(drop=True)
    leak_checks(h, quintes)
    pred = predictions(h, quintes, O / "predictions.csv")
    table, detail = evaluate(pred, quintes)
    detail.to_csv(O / "payouts_par_course.csv")
    table.round(2).to_csv(O / "resultats.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(1).to_string(index=False))
    for bet in ["Gagnant", "Placé"]:
        m, lo, hi = paired_diff(detail, f"{bet}|2e du modèle", f"{bet}|2e de la cote")
        print(f"{bet} : 2e modèle − 2e cote = {m:+.1f} pts de ROI, IC95 [{lo:+.1f} ; {hi:+.1f}]")
    print("Modèle 2 = Cote 2 dans", f"{(pred.model_2 == pred.market_2).mean():.0%}", "des quintés ;",
          "Modèle 1 = Cote 1 dans", f"{(pred.model_1 == pred.market_1).mean():.0%}")
