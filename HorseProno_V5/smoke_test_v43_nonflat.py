import json
import pandas as pd
from portable_lgbm import PortableBooster
from quinte_v43 import load_outsider_artifact, rank_quinte_v43
h=pd.read_csv('validated_history.csv');h['race_date']=pd.to_datetime(h['race_date']);h['is_non_runner']=h['is_non_runner'].fillna(False).astype(bool)
rid='R1C7_2026-07-11';race=h[(h.race_id==rid)&(~h.is_non_runner)].copy();day=race.race_date.iloc[0];history=h[(h.race_date<day)&(~h.is_non_runner)].copy()
artifact=json.load(open('quinte_v2_artifact.json'));ranker=PortableBooster('quinte_v2_ranker.txt');od=load_outsider_artifact('v43_outsider_artifact.json')
out=rank_quinte_v43(history,race,artifact,ranker,od);assert len(out[out.selected_top7])==7; assert out.iloc[0].v43_mode=='V42_BASE'; assert not out.v43_outsider_candidate.any();print('OK - V4.3 hors PLAT conserve V4.2')
