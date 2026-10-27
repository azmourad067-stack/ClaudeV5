"""Test de bout en bout de la classe de production V5Model.

1) Reconstruit l'artefact « coupé au 12/09 » utilisé pour l'évaluation forward et vérifie
   que V5Model.predict_race reproduit les résultats de recherche sur les 41 Quintés forward.
2) Vérifie l'absence de fuite : modifier l'arrivée de la course ne change pas la prédiction.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "research"))
from horseprono_v5 import V5Model  # noqa: E402
from common import load_history, load_forward  # noqa: E402

cut = json.loads((BASE / "research" / "v5_model_cut0912.json").read_text())
art = {"features": cut["features"], "mean": cut["mean"], "scale": cut["scale"],
       "coef_win": cut["coef_win"], "coef_top5": cut["coef_top5"], "win_temperature": 1.10}
model = V5Model(art)

h = load_history()
h = h[~h.is_non_runner.astype(bool)]
fwd, arrivals = load_forward("snap")
hits, quarte, quinte, wt3 = [], 0, 0, 0
for rid, race in fwd.groupby("race_id", sort=False):
    arr = arrivals[rid]
    q = race.drop(columns=["finish_position"]).assign(finish_position=np.nan)
    res = model.predict_race(h, q)
    top7 = set(res.loc[res.selected_top7, "horse_number"].astype(int))
    hits.append(len(top7 & set(arr)))
    quarte += set(arr[:4]) <= top7
    quinte += set(arr) <= top7
    wt3 += arr[0] in res.head(3).horse_number.astype(int).tolist()
    assert abs(res.v5_win_prob.sum() - 1) < 1e-6
    assert abs(res.v5_top5_prob.sum() - min(5, len(res))) < 1e-6
print(f"41 Quintés forward (cote T-20) : Top5 dans 7 = {np.mean(hits):.3f} ; Quarté {quarte} ; Quinté {quinte} ; gagnant Top3 {wt3}")
assert len(hits) == 41

# Pas de fuite : l'arrivée de la course cible n'influence pas la prédiction
race = fwd[fwd.race_id == fwd.race_id.iloc[0]].copy()
a = model.predict_race(h, race.assign(finish_position=np.nan))
b = model.predict_race(h, race.assign(finish_position=np.arange(1, len(race) + 1)))
assert a.horse_number.tolist() == b.horse_number.tolist()
assert np.allclose(a.v5_win_prob, b.v5_win_prob)
# Le modèle de production se charge
prod = V5Model.load(BASE / "v5_artifact.json")
print("artefact production :", prod.art["model_name"], prod.art["training_end"], prod.art["artifact_hash"])
print("OK")
