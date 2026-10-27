# HorseProno V5 — modèle probabiliste Plackett-Luce

Main file Streamlit : `streamlit_app.py` (inchangé). Aucune nouvelle dépendance à l'exécution
(numpy/pandas). `scipy` n'est nécessaire que pour réentraîner (`train_v5.py`).

## Ce qui change

| | V4.5 | V5 |
|---|---|---|
| Données d'apprentissage | 98 Quintés (902 lignes) | toutes les courses : 5 023 courses, 56 000 lignes |
| Structure | 2 modèles + 6 couches de règles et poids réglés à la main sur 58 courses | 1 jeu de features, 2 têtes de vraisemblance (gagnant ; arrivée 1→5) |
| Musique PMU | non utilisée (V4.3 : regex cassée, toujours 0) | 12 variables (forme 3/6 courses, échecs D/A/T, spécialité, absence) |
| Sortie | rangs et scores | P(gagner), P(Top3), P(Top5), cote juste |
| Temps de calcul | ~12 s / course | ~2-3 s / course |

Fichiers :
- `horseprono_v5.py` — features, entraînement (logit conditionnel « exploded »), inférence.
- `train_v5.py` — produit `v5_artifact.json` à partir de `validated_history.csv`.
- `v5_artifact.json` — modèle entraîné sur le 01/06 → 16/09/2026.
- `v5_evaluation_report.json` — résultats hors échantillon (affichés dans l'app).
- `smoke_test_v5.py` — rejoue les 41 Quintés forward et vérifie l'absence de fuite.
- `research/` — scripts d'évaluation reproductibles ; `data/` — extraits Supabase utilisés.

L'ancien moteur V4.5 reste disponible dans l'app (case « Afficher aussi le Top 7 de l'ancien moteur »).

## Réentraînement

```bash
python train_v5.py                      # tout validated_history.csv
python smoke_test_v5.py                 # contrôle
```

Prérequis bloquant : les arrivées (`participants.finish_position`) ne sont plus écrites dans
Supabase depuis le 16/09/2026, alors que les rapports PMU sont bien collectés dans
`quinte_payouts_raw`. Tant que ce n'est pas réparé, l'historique interne du modèle reste figé.
