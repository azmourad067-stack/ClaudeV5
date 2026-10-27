# HorseProno Quinté V3 — Streamlit

## Déploiement Streamlit Community Cloud

1. Décompresse tout le ZIP à la racine de ton dépôt GitHub.
2. Vérifie que `streamlit_app.py`, `quinte_v3.py`, `quinte_v2.py`, `quinte_candidate.py`, `quinte_v2_artifact.json`, `quinte_v2_ranker.txt` et `validated_history.csv` sont tous à la racine.
3. Dans Streamlit Community Cloud, choisis ton dépôt et ta branche.
4. **Main file path : `streamlit_app.py`**.
5. Déploie.

Les secrets Supabase sont optionnels pour faire une prédiction. Ils servent au complément d'historique et aux snapshots de marché :

```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "..."              # lecture, optionnel
SUPABASE_SERVICE_KEY = "..."      # écriture snapshot, optionnel et à garder uniquement dans Streamlit Secrets
```

Ne mets jamais `SUPABASE_SERVICE_KEY` dans GitHub.

## Sortie V3

- 4 `NOYAU_V3`
- 1 `CHALLENGER_RANKER`
- 1 `CHALLENGER_STABILISATEUR`
- 1 `CHALLENGER_EDGE_PROFOND`

Le programme PMU complet est chargé automatiquement ; l'utilisateur choisit ensuite la réunion et la course.
