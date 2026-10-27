# HorseProno Quinté V3 — Streamlit Cloud Safe

## Déploiement
- Main file path: `streamlit_app.py`
- Python recommandé: **3.12** (valeur par défaut de Streamlit Community Cloud)
- `requirements.txt` contient uniquement les dépendances Python nécessaires.
- `packages.txt` installe `libgomp1`, utilisé par LightGBM sous Linux.

## Supabase (optionnel)
Le SDK Python Supabase n'est plus nécessaire. L'application utilise directement la Data API REST via `requests`.

Secrets acceptés :
- `SUPABASE_URL`
- lecture: `SUPABASE_PUBLISHABLE_KEY` ou ancien `SUPABASE_KEY`
- écriture snapshots: `SUPABASE_SECRET_KEY` ou ancien `SUPABASE_SERVICE_KEY`

L'app fonctionne sans secrets Supabase avec l'historique CSV embarqué.


## V3 Portable Ranker

Cette version n'exige plus le package Python `lightgbm` sur Streamlit Community Cloud.
Le fichier entraîné `quinte_v2_ranker.txt` est lu par `portable_lgbm.py`, un moteur d'inférence
Python pur validé numériquement contre LightGBM 4.6.0 (mêmes prédictions sur le modèle fourni).
Le modèle et la stratégie 4+3 ne sont donc pas modifiés.
