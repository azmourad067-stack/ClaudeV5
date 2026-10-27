# HorseProno Quinté V4.1 — PLAT spécialisé

## Main file Streamlit
`streamlit_app.py`

## Principe
V4.1 conserve l'architecture de fusion V2/V3 : les chevaux communs aux deux Top7 sont conservés, puis les places de désaccord sont arbitrées.

### En PLAT
Score d'arbitrage spécialisé :
- 0.60 × percentile modèle Top5
- 0.20 × percentile LambdaMART
- 0.10 × contexte
- 0.80 × percentile marché (pénalité)

Cette recette rend l'arbitrage moins dépendant du marché sur les Quintés de plat.

### Hors PLAT
La recette V4 générale est conservée :
- 0.60 × modèle Top5
- 0.25 × LambdaMART
- 0.15 × contexte
- 0.40 × marché (pénalité)

## Protocole historique PLAT
- 31 Quintés PLAT dans le walk-forward.
- 17 premières courses : développement de la recette PLAT.
- 14 suivantes : contrôle temporel sans modification de la recette.

Résultats PLAT sur 31 courses :
- V4 générale : 11 Trio / 8 Quarté / 4 Quinté ; 3.290 chevaux de l'arrivée /5 en moyenne.
- V4.1 PLAT : 12 Trio / 9 Quarté / 4 Quinté ; 3.323/5 en moyenne.

Sur les 14 courses de contrôle temporel, V4.1 conserve les mêmes résultats principaux que V4 : 5 Trio / 4 Quarté / 2 Quinté et 3.214/5.

Important : le corpus global a déjà été consulté pendant le développement V2/V3/V4. La promotion définitive doit se faire sur de nouveaux Quintés en validation forward.

## Déploiement
Décompresser le ZIP à la racine du dépôt GitHub et utiliser `streamlit_app.py` comme Main file path dans Streamlit Community Cloud.
