# HorseProno Quinté V4 — Candidate Fusion V2/V3

V4 conserve tous les chevaux communs aux Top7 V2 et V3, puis remplit les places restantes parmi leurs désaccords avec un score edge pré-course :

`0.60*model_rankpct + 0.25*ranker_percentile + 0.15*context_score - 0.40*market_rankpct`

## Résultats rétrospectifs — 58 Quintés walk-forward

- V2 : Trio 24/58, Quarté 14/58, Quinté 5/58, moyenne 3.293 chevaux du Top5 dans les 7.
- V3 : Trio 22/58, Quarté 13/58, Quinté 5/58, moyenne 3.276.
- V4 candidate : Trio 23/58, Quarté 15/58, Quinté 9/58, moyenne 3.328.

Sur les 28 dernières courses du bloc :
- V2 : Trio 10/28, Quarté 6/28, Quinté 3/28, moyenne 3.25.
- V3 : Trio 8/28, Quarté 6/28, Quinté 2/28, moyenne 3.179.
- V4 : Trio 10/28, Quarté 8/28, Quinté 5/28, moyenne 3.286.

Ces résultats sont rétrospectifs. Le bloc historique a déjà servi au développement des V2/V3/V4 ; la V4 doit maintenant être validée en forward sur de nouveaux Quintés.

## Streamlit

Main file path : `streamlit_app.py`

Le moteur portable LightGBM de la V3 est conservé. Si le package `lightgbm` n'est pas disponible, `portable_lgbm.py` charge le même modèle directement.
