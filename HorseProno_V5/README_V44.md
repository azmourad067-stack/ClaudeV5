# HorseProno Quinté V4.4 — Winner Top 3

V4.4 ajoute un objectif séparé : **faire remonter le gagnant officiel dans les 3 premières propositions**, sans toucher aux 7 chevaux sélectionnés par V4.2/V4.3.

## PLAT

Le Top7 est réordonné avec :

- 50 % rang relatif du modèle Top5
- 40 % rang relatif LambdaMART
- 10 % moyenne forme / régularité / contexte
- 0 % marché direct

Hors PLAT, l'ordre V4.3 est conservé.

## Rétro-analyse walk-forward

Sur 58 Quintés :
- gagnant Top3 : 30/58 -> **31/58**
- gagnant Top2 : 20/58 -> **22/58**
- gagnant Top1 : 15/58 -> **15/58**

Sur 31 Quintés PLAT :
- gagnant Top3 : 12/31 -> **13/31**
- gagnant Top2 : 5/31 -> **7/31**
- gagnant Top1 : 3/31 -> **3/31**

Sur les 14 PLAT du bloc de contrôle :
- Top3 : 4/14 -> **5/14**
- Top2 : 2/14 -> **2/14**
- Top1 : 2/14 -> **2/14**

La shortlist de 7 chevaux est strictement identique à V4.2/V4.3 : les métriques Trio/Quarté/Quinté **dans les 7** ne peuvent donc pas être dégradées par V4.4.

`v44_backtest_58.csv` contient le détail course par course.
