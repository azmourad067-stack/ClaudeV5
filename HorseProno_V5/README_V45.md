# HorseProno Quinté V4.5 — Market Dynamics

V4.5 répond au constat suivant : le classement peut changer fortement entre T-30, T-2 et l'après-course parce que les cotes PMU évoluent. Une cote observée après le départ ne doit jamais être utilisée comme information pré-course ; V4.5 cherche donc à **projeter le mouvement probable vers T-2** à partir d'informations disponibles avant le départ.

## Ce qui ne change pas

- Le Top7 est toujours construit par V4.2/V4.3.
- V4.3 Outsider Detector reste en shadow mode.
- V4.5 ne remplace aucun cheval du Top7 : elle modifie uniquement l'ordre Winner Top3 sur le PLAT.

## Signal V4.5

Le coefficient de dynamique combine :

1. **Momentum** : cote projetée en baisse = signal positif ; cote projetée en hausse = signal négatif.
2. **Volatilité** : un mouvement irrégulier réduit la confiance.
3. **Confiance** :
   - un seul instant : la différence PMU `dernierRapportDirect / dernierRapportReference` sert de proxy conservateur ;
   - 2 snapshots live : une pente de `log(cote)` prend le relais ;
   - 3 snapshots ou plus : la confiance augmente si la trajectoire est régulière.

Le coefficient ajouté au score Winner V4.4 est borné à **±0,10**.

## Important : pas de fuite temporelle

Si l'analyse est lancée après le départ, Streamlit affiche un avertissement. Les données post-départ peuvent être enregistrées pour l'étude future, mais elles ne doivent pas être utilisées comme preuve d'une performance prédictive pré-course.

## Snapshots

La session Streamlit garde automatiquement les snapshots de la course ouverte. Pour les conserver entre les sessions et construire un vrai jeu d'entraînement T-30/T-2, configurez côté serveur :

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_KEY = "<publishable/anon read key>"
SUPABASE_SECRET_KEY = "<server-side secret/service-role key>"
```

Ne jamais committer la clé `SUPABASE_SECRET_KEY` dans GitHub.

La table `public.quinte_market_snapshots` reçoit désormais : `scheduled_start`, `minutes_to_start`, `odds_direct`, `odds_reference`, `trend_indicator`, `trend_strength`, `projected_odds_t2`, `predicted_log_move`, `volatility_score`, `dynamics_confidence`, `dynamics_score`, `model_rank`, `model_name`.

## Validation prévue

Après accumulation d'un nombre suffisant de Quintés PLAT avec snapshots T-30/T-10/T-2, remplacer le coefficient proxy par un coefficient appris uniquement sur les snapshots pré-course. L'évaluation doit rester temporelle : jours anciens pour l'apprentissage, jours plus récents pour le contrôle.
