# HorseProno Quinté V4.3 — Outsider Detector (shadow mode)

## Principe

La V4.3 conserve **exactement le Top7 V4.2** comme sélection officielle.
Sur les courses de **PLAT**, elle analyse en parallèle tous les chevaux absents des Top7 V2 et V3.
Elle affiche :

- le meilleur outsider hors V2/V3 ;
- son score de détection ;
- le cheval V4.2 qu'il aurait théoriquement remplacé ;
- si le seuil expérimental de 0,50 est franchi.

Aucun remplacement n'est appliqué automatiquement dans cette version. Le but est de collecter des prédictions forward propres avant d'autoriser le wildcard.

## Validation temporelle

Le détecteur a été conçu sur les 17 premiers Quintés PLAT du bloc walk-forward, puis contrôlé sur les 14 suivants.

- V4.2 développement : 3,471 chevaux Top5 captés /5 ; 8 Trios ; 5 Quartés ; 2 Quintés.
- V4.3 détecteur en validation croisée développement : 3,529/5 ; mêmes 8 Trios, 5 Quartés, 2 Quintés ; 3 signaux, 1 amélioration, 0 dégradation.
- V4.2 contrôle 14 courses : 3,286/5 ; 5 Trios ; 4 Quartés ; 2 Quintés.
- V4.3 au seuil 0,50 sur le contrôle : 0 remplacement ; résultats identiques à V4.2.
- AUC outsider sur le contrôle : ~0,51. Le signal n'est donc pas encore assez stable pour un remplacement automatique.

## Déploiement Streamlit

Main file path : `streamlit_app.py`

Le package n'a pas besoin de `lightgbm` ni de `scikit-learn` à l'exécution. Le ranker utilise `portable_lgbm.py` si LightGBM n'est pas installé et le détecteur outsider utilise les coefficients présents dans `v43_outsider_artifact.json`.
