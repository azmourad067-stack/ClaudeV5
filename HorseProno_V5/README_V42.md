# HorseProno Quinté V4.2 — PLAT contextuel

V4.2 conserve la fusion V2/V3 et ne modifie que l'arbitrage des désaccords sur les courses de PLAT.

## Règle PLAT

Base : `0.60 modèle Top5 + 0.20 LambdaMART + 0.10 contexte`.

Le coefficient marché dépend de l'écart de poids pré-course :

- écart `<= 6 kg` : `-0.80`
- écart `> 6 et <= 8 kg` : `-1.00`
- écart `> 8 kg` : `-0.50`
- poids incomplets : fallback `-0.80`

L'écart de poids est un **proxy** de structure du handicap. Le dataset ne contient pas de champ explicite indiquant si la course est un handicap.

## Résultats PLAT

Sur 31 Quintés PLAT du walk-forward :

- V4.1 : 3.323 chevaux de l'arrivée /5, 12 Trios, 9 Quartés, 4 Quintés.
- V4.2 : **3.387 /5, 13 Trios, 9 Quartés, 4 Quintés**.

Contrôle temporel sur les 14 PLAT les plus récents :

- V4.1 : 3.214 /5, 5 Trios, 4 Quartés, 2 Quintés.
- V4.2 : **3.286 /5, 5 Trios, 4 Quartés, 2 Quintés**.

Sur les 58 Quintés toutes disciplines, les autres disciplines restant en V4 générale :

- V4.1 : 3.345 /5, 24 Trios, 16 Quartés, 9 Quintés.
- V4.2 : **3.379 /5, 25 Trios, 16 Quartés, 9 Quintés**.

## Ce qui n'a PAS été ajouté

- Pas de coefficient par hippodrome : échantillons trop faibles.
- Pas de correction directe de corde : le diagnostic existe, mais le bonus testé dégradeait le backtest.
- Pas de règle 17+ partants : seulement deux courses disponibles; l'app affiche simplement une alerte de confiance.
- Pas de règle de distance dans le score : le segment 1500–2000 m est le plus régulier, mais la stabilité n'est pas encore suffisante pour créer une pondération dédiée.

## Streamlit

Main file path : `streamlit_app.py`.

V4.2 utilise le ranker portable; aucune installation LightGBM n'est requise.
