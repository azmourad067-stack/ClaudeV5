# 2e du modèle HorseProno V5 au simple gagnant / placé — quintés, 01/11/2025 → 09/11/2026

## Verdict : non rentable
374 quintés, 1 € par course, réglés au rapport définitif officiel (aucune course exclue).

| Pari | Stratégie | Paris | Réussite | ROI | IC 95 % (bootstrap) | 1re moitié | 2e moitié |
|---|---|---|---|---|---|---|---|
| Gagnant | **2e du modèle** | 374 | 15,8 % | **−10,5 %** | [−32,1 ; +12,2] | +13,0 % | −34,0 % |
| Gagnant | 2e de la cote | 374 | 14,2 % | −26,1 % | [−45,2 ; −6,0] | −44,2 % | −8,1 % |
| Gagnant | Favori du modèle | 374 | 25,7 % | −11,8 % | [−28,0 ; +5,4] | −22,4 % | −1,2 % |
| Gagnant | Favori de la cote | 374 | 28,6 % | +3,3 % | [−14,4 ; +22,2] | +8,7 % | −2,1 % |
| Placé | **2e du modèle** | 374 | 40,4 % | **−9,8 %** | [−21,2 ; +2,2] | −5,8 % | −13,9 % |
| Placé | 2e de la cote | 374 | 38,2 % | −17,0 % | [−28,0 ; −5,7] | −25,7 % | −8,2 % |
| Placé | Favori du modèle | 374 | 51,6 % | −6,4 % | [−15,9 ; +3,4] | −6,1 % | −6,6 % |
| Placé | Favori de la cote | 374 | 53,2 % | −4,5 % | [−14,0 ; +4,9] | +1,2 % | −10,3 % |

Moitiés : 01/11/2025 → 06/05/2026 et 07/05/2026 → 09/11/2026 (187 quintés chacune).

- Critère (ROI > 0 sur les deux moitiés ET IC 95 % > 0) : **échoue** au gagnant (+13 % puis −34 %, IC à cheval sur 0) et au placé (négatif partout).
- 2e modèle − 2e cote (apparié, mêmes courses) : +15,7 pts au gagnant, IC [−5,9 ; +37,7] ; +7,1 pts au placé, IC [−4,3 ; +18,3]. Non significatif.
- Le 2e du modèle est le 2e de la cote dans 57 % des quintés ; le favori, dans 82 %.
- Aucune stratégie n'a d'IC entièrement au-dessus de 0. Le +65 % observé sur nov.–janv. (analyse intermédiaire) était du bruit.

## Protocole sans fuite
- **Modèle** : code du dépôt (`horseprono_v5.py` : features `FEATURES_PROD`, logit exploded Plackett-Luce,
  inférence `V5Model.predict_race`). `v5_artifact.json` (appris jusqu'au 16/09/2026) **n'est pas utilisé**.
  Réentraînement chaque lundi sur les seules courses de date < lundi (54 modèles, 2 114 → 15 456 courses).
- **Historique** : courses françaises du 01/09/2025 au 09/11/2026 (API PMU), places 1..5 comme dans ClaudeV5.
  Stats cheval/jockey/entraîneur et forme calculées sur des dates strictement antérieures.
- **Cote** : `dernierRapportReference`, gardée seulement si publiée avant le départ (médiane : 30 min avant).
  `dernierRapportDirect` n'est jamais utilisée : après la course elle devient la cote finale.
- **Musique** : 1er symbole = arrivée du jour dans 11 % des cas, contre 96 % pour la musique de la course
  suivante → la musique lue est bien pré-course.
- **Contrôle direct** (quinté du 10/11/2026, `snapshot_live.py`) : après l'arrivée, la cote directe change
  (ex. n°2 : 25 → 18), la cote de référence, la musique et le statut sont inchangés.

## Pourquoi le classement de l'application change après la course
`streamlit_app.py` (l. 513) utilise `dernierRapportDirect` en priorité. Après le départ, c'est la cote
finale : V5 étant dominé par la cote, son classement change. Sur les 41 quintés de `data/` (cote T-20 vs cote
en base, plus tardive), le 2e de V5 change dans 11 cas et trouve 10 gagnants au lieu de 8. Tout backtest
calculé avec une cote relue après la course est donc optimiste. Correctif conseillé : figer la cote au moment
du pari (snapshot horodaté) et ne jamais relire `dernierRapportDirect` pour une course partie.

## Limites
- ~1 quinté/jour : sur 187 courses, un ROI de ±20–30 % sort par hasard (voir largeur des IC).
- Cote à T-30 (référence) et non T-20 (directe) comme dans l'appli : le 2e de la cote coïncide dans 16/41 quintés
  entre les deux. Le backtest mesure « V5 avec la cote T-30 ».
- Historique limité aux courses françaises (ClaudeV5 incluait l'étranger) et démarrant 2 mois avant le 1er quinté.
- Non-partants pris au statut final ; rapports « placé » officiels (les e-rapports internet diffèrent légèrement).

## Reproduire
```bash
export PMU_CACHE=/chemin/cache
python collect_pmu.py            # ~4–5 h, 1 requête/s, reprenable
python build_dataset.py
python backtest.py               # ~25 min → results/
```
