import json
import pandas as pd
from portable_lgbm import PortableBooster
from quinte_v43 import load_outsider_artifact, rank_quinte_v43
from quinte_v44 import rank_quinte_v44

h=pd.read_csv('validated_history.csv')
h['race_date']=pd.to_datetime(h['race_date'])
h['is_non_runner']=h['is_non_runner'].fillna(False).astype(bool)
artifact=json.load(open('quinte_v2_artifact.json'))
ranker=PortableBooster('quinte_v2_ranker.txt')
od=load_outsider_artifact('v43_outsider_artifact.json')

# PLAT: membership Top7 identique, ordre spécialisé actif
rid='R1C3_2026-09-06'
race=h[(h.race_id==rid)&(~h.is_non_runner)].copy(); day=race.race_date.iloc[0]
history=h[(h.race_date<day)&(~h.is_non_runner)].copy()
v43=rank_quinte_v43(history,race,artifact,ranker,od)
v44=rank_quinte_v44(history,race,artifact,ranker,od)
set43=set(v43[v43.selected_top7].horse_number.astype(int)); set44=set(v44[v44.selected_top7].horse_number.astype(int))
assert set43==set44 and len(set44)==7
assert v44.iloc[0].v44_mode=='WINNER_TOP3_PLAT'
assert v44.head(3).v44_winner_score.notna().all()

# Hors PLAT: ordre strictement identique à V4.3
rid2='R1C7_2026-07-11'
race2=h[(h.race_id==rid2)&(~h.is_non_runner)].copy(); day2=race2.race_date.iloc[0]
history2=h[(h.race_date<day2)&(~h.is_non_runner)].copy()
a=rank_quinte_v43(history2,race2,artifact,ranker,od)
b=rank_quinte_v44(history2,race2,artifact,ranker,od)
assert a.horse_number.astype(int).tolist()==b.horse_number.astype(int).tolist()
assert b.iloc[0].v44_mode=='V43_ORDER'

print('OK - HorseProno Quinté V4.4 Winner Top 3')
print('PLAT Top7 inchangé:', '-'.join(map(str, v44.head(7).horse_number.astype(int).tolist())))
print('PLAT Top3 V4.4:', '-'.join(map(str, v44.head(3).horse_number.astype(int).tolist())))
print('Hors PLAT: ordre V4.3 conservé')
