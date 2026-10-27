import json
import pandas as pd
from portable_lgbm import PortableBooster
from quinte_v42 import rank_quinte_v42

h = pd.read_csv('validated_history.csv')
h['race_date'] = pd.to_datetime(h['race_date'])
h['is_non_runner'] = h['is_non_runner'].fillna(False).astype(bool)
rid = 'R1C3_2026-09-06'
race = h[(h['race_id'] == rid) & (~h['is_non_runner'])].copy()
day = race['race_date'].iloc[0]
history = h[(h['race_date'] < day) & (~h['is_non_runner'])].copy()
artifact = json.load(open('quinte_v2_artifact.json'))
ranker = PortableBooster('quinte_v2_ranker.txt')
out = rank_quinte_v42(history, race, artifact, ranker)
sel = out[out['selected_top7']]
assert len(sel) == 7
assert out.iloc[0]['v42_mode'] == 'PLAT_CONTEXTUEL'
print('OK - HorseProno Quinté V4.2')
print('Top7:', '-'.join(sel['horse_number'].astype(int).astype(str)))
print('Ecart poids:', out.iloc[0]['v42_weight_spread'])
print('Coeff marché:', out.iloc[0]['v42_market_weight'])
