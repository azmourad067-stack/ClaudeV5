"""Entraîne l'artefact HorseProno V5 (deux têtes logit conditionnel) à partir de l'historique validé.

Usage :
    python train_v5.py                                 # validated_history.csv, toutes dates
    python train_v5.py --history autre.csv --end 2026-10-31

L'historique doit contenir, pour chaque course retenue, une arrivée 1..5 complète.
Réentraîner dès que de nouvelles arrivées sont disponibles (aujourd'hui : la base
Supabase n'a plus de `finish_position` après le 16/09/2026, voir RAPPORT).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from horseprono_v5 import FEATURES_PROD, build_features, fit_exploded_logit

BASE = Path(__file__).resolve().parent

# Estimée chronologiquement sur les courses du 12 au 30/09/2026 (cote T-20),
# vérifiée sur octobre : courbe de calibration nettement meilleure sur les favoris
# (prédit 0,63 -> 0,585 pour 0,52 observé). Ne modifie aucun classement.
WIN_TEMPERATURE = 1.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(BASE / "validated_history.csv"))
    ap.add_argument("--end", default=None, help="date exclue (AAAA-MM-JJ) ; défaut : tout l'historique")
    ap.add_argument("--out", default=str(BASE / "v5_artifact.json"))
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    h = pd.read_csv(args.history, low_memory=False)
    h["race_date"] = pd.to_datetime(h["race_date"]).dt.normalize()
    h = h[~h["is_non_runner"].fillna(False).astype(bool)].reset_index(drop=True)
    if args.end:
        h = h[h.race_date < pd.Timestamp(args.end)].reset_index(drop=True)

    feats = build_features(h, h)
    X = feats[FEATURES_PROD].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    g = h["race_id"].astype(str).to_numpy()
    pos = pd.to_numeric(h["finish_position"], errors="coerce").fillna(99).astype(int).to_numpy()
    coef_win = fit_exploded_logit(Z, g, pos, k=1, l2=args.l2)
    coef_top5 = fit_exploded_logit(Z, g, pos, k=5, l2=args.l2)

    art = {
        "model_name": "HorseProno_V5_PlackettLuce",
        "package_version": "5.0.0",
        "method": "Logit conditionnel (gagnant) + logit exploded Plackett-Luce (places 1..5), toutes courses",
        "features": FEATURES_PROD,
        "mean": mu.tolist(),
        "scale": sd.tolist(),
        "coef_win": coef_win.tolist(),
        "coef_top5": coef_top5.tolist(),
        "win_temperature": WIN_TEMPERATURE,
        "l2": args.l2,
        "training_races": int(h.race_id.nunique()),
        "training_rows": int(len(h)),
        "training_start": str(h.race_date.min().date()),
        "training_end": str(h.race_date.max().date()),
        "standardised_coefficients": {
            "win": dict(zip(FEATURES_PROD, np.round(coef_win, 4).tolist())),
            "top5": dict(zip(FEATURES_PROD, np.round(coef_top5, 4).tolist())),
        },
    }
    art["artifact_hash"] = hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()[:16]
    Path(args.out).write_text(json.dumps(art, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{args.out} : {art['training_races']} courses, {art['training_start']} -> {art['training_end']}")


if __name__ == "__main__":
    main()
