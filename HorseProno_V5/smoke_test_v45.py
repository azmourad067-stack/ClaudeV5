from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from quinte_v2 import load_artifact
from quinte_v3 import load_ranker
from quinte_v43 import load_outsider_artifact
from quinte_v44 import rank_quinte_v44
from quinte_v45 import apply_v45_market_dynamics

B=Path(__file__).resolve().parent
hist=pd.read_csv(B/'validated_history.csv', low_memory=False)
hist['race_date']=pd.to_datetime(hist['race_date'], errors='coerce')
# une course PLAT historique suffisamment récente du fichier
rids=[]
for rid,g in hist.groupby('race_id'):
    if str(g.iloc[0].get('discipline','')).upper()=='PLAT' and g['odds'].notna().sum()>=7:
        rids.append((g['race_date'].max(),rid))
rids.sort(reverse=True)
rid=rids[min(5,len(rids)-1)][1]
g=hist[hist.race_id==rid].copy().reset_index(drop=True)
day=g['race_date'].iloc[0]
history=hist[hist['race_date']<day].copy()
race=g.copy()
race['finish_position']=pd.NA
# ajoute des cotes direct/reference synthétiques cohérentes
race['odds_direct']=pd.to_numeric(race['odds'], errors='coerce')
race['odds_reference']=race['odds_direct']*1.08
race['trend_indicator']=''
race['trend_strength']=pd.NA
artifact=load_artifact(B/'quinte_v2_artifact.json')
ranker=load_ranker(B/'quinte_v2_ranker.txt')
outs=load_outsider_artifact(B/'v43_outsider_artifact.json')
v44=rank_quinte_v44(history,race,artifact,ranker,outs)
now=datetime.now(timezone.utc)
snaps={}
for row in race.itertuples(index=False):
    if pd.notna(row.odds_direct):
        # deux observations : certains chevaux se contractent, d'autres dérivent
        factor=1.10 if int(row.horse_number)%2==0 else 0.92
        snaps[int(row.horse_number)]=[
            {'captured_at':(now-timedelta(minutes=20)).isoformat(),'odds':float(row.odds_direct)*factor},
            {'captured_at':now.isoformat(),'odds':float(row.odds_direct)},
        ]
v45=apply_v45_market_dynamics(v44,snaps,minutes_to_start=10.0)
set44=set(v44.head(7).horse_number.astype(int))
set45=set(v45.head(7).horse_number.astype(int))
assert set44==set45, (set44,set45)
assert 'v45_dynamics_score' in v45.columns
assert v45['v45_dynamics_confidence'].between(0,1).all()
print('OK - HorseProno Quinté V4.5')
print('Course test:',rid)
print('Top7 inchangé:', '-'.join(map(str,v45.head(7).horse_number.astype(int))))
print('Top3 V44:', '-'.join(map(str,v44.head(3).horse_number.astype(int))))
print('Top3 V45:', '-'.join(map(str,v45.head(3).horse_number.astype(int))))
print(v45.head(7)[['horse_number','odds','odds_reference','v45_projected_odds_t2','v45_dynamics_score','v45_dynamics_confidence','v45_adjustment']].to_string(index=False))
