import json
import pandas as pd
from portable_lgbm import PortableBooster
from quinte_v43 import load_outsider_artifact, rank_quinte_v43
h=pd.read_csv('validated_history.csv');h['race_date']=pd.to_datetime(h['race_date']);h['is_non_runner']=h['is_non_runner'].fillna(False).astype(bool)
rid='R1C3_2026-09-06';race=h[(h.race_id==rid)&(~h.is_non_runner)].copy();day=race.race_date.iloc[0];history=h[(h.race_date<day)&(~h.is_non_runner)].copy()
artifact=json.load(open('quinte_v2_artifact.json'));ranker=PortableBooster('quinte_v2_ranker.txt');od=load_outsider_artifact('v43_outsider_artifact.json')
out=rank_quinte_v43(history,race,artifact,ranker,od);sel=out[out.selected_top7]
assert len(sel)==7
assert out.iloc[0].v43_mode=='OUTSIDER_SHADOW'
outs=out[out.v43_outsider_candidate]
assert len(outs)==1
print('OK - HorseProno Quinté V4.3 shadow')
print('Top7 V4.2:', '-'.join(sel.horse_number.astype(int).astype(str)))
print('Outsider V4.3:', int(outs.iloc[0].horse_number), 'score=', round(float(outs.iloc[0].v43_outsider_probability),4))
print('Threshold met:', bool(outs.iloc[0].v43_shadow_threshold_met))
print('Replacement theoretical:', out.iloc[0].v43_shadow_replacement_number)
