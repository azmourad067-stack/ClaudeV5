# HorseProno V5 — Streamlit

Main file path : `streamlit_app.py`

V5 remplace l'empilement V4.2→V4.5 par un modèle probabiliste unique (Plackett-Luce) entraîné sur toutes les courses. Voir `README_V5.md` et `RAPPORT_V5.md`.

## Déploiement

1. Déposer tous les fichiers du dossier à la racine du dépôt GitHub.
2. Dans Streamlit Community Cloud, choisir `streamlit_app.py` comme Main file path.
3. Les secrets Supabase sont optionnels pour lancer le moteur. Pour conserver les snapshots entre les sessions, ajouter dans les Secrets Streamlit :

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_KEY = "<publishable/anon read key>"
SUPABASE_SECRET_KEY = "<server-side secret/service-role key>"
```

Ne jamais committer `.streamlit/secrets.toml` ou la clé serveur dans GitHub.

Voir `README_V5.md` pour le fonctionnement ; `supabase_v45_market_dynamics.sql` décrit toujours la table de snapshots.
