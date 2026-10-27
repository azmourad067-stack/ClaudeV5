"""Candidat de recherche top-5. Même préparation pour apprentissage et prédiction.
Les résultats du jour sont exclus, même si plusieurs chevaux partagent un entraîneur.
Score de classement, pas une probabilité calibrée ni un rapport PMU.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def music_score(value):
    """Parse les zéros/incidents; retire les années entre parenthèses."""
    clean = re.sub(r'\(\d{2,4}\)', '', str(value or ''))
    tokens = re.findall(r'(\d+|[DATdat])\s*[apmhsAPMHS]', clean)
    vals = [max(0., 1-(int(t)-1)/9) if t.isdigit() and int(t)>0 else 0. for t in tokens[:5]]
    return float(np.mean(vals)) if vals else np.nan


def key(s):
    return s.fillna('').astype(str).str.strip().str.upper()


def prepare(history, query):
    """history: courses terminées, arrivées 1..5 validées, non-partants exclus.
    query: partants à classer, toutes les lignes de chaque course nécessaires.
    Aucun résultat de query n'est lu. history peut inclure des dates futures,
    qui sont exclues par la jointure strictement antérieure.
    Les rangs manquants de history sont hors top5 UNIQUEMENT parce que
    l'appelant a vérifié une arrivée 1..5 complète et unique.
    """
    h, q = history.copy(), query.copy().reset_index(drop=True)
    h['race_date'] = pd.to_datetime(h.race_date).dt.normalize()
    q['race_date'] = pd.to_datetime(q.race_date).dt.normalize()
    out = pd.DataFrame(index=q.index)
    n = q.groupby('race_id').horse_number.transform('size').clip(lower=2)
    odds = pd.to_numeric(q.odds, errors='coerce').where(lambda x: x>1)
    inv = 1/odds
    out['log_odds'] = np.log(odds.fillna(20))
    out['market_prob'] = (inv / inv.groupby(q.race_id).transform('sum')).fillna(1/n)
    out['odds_missing'] = odds.isna().astype(float)
    out['field_size'] = n
    out['distance'] = pd.to_numeric(q.distance, errors='coerce').fillna(2000)/1000
    for c,default in [('draw',0),('weight',57),('age',5)]:
        v = pd.to_numeric(q[c], errors='coerce')
        out[c+'_missing'] = v.isna().astype(float)
        out[c] = v.fillna(default)
    out['draw'] = out['draw']/n
    out['weight'] = (out.weight-out.weight.groupby(q.race_id).transform('median'))/5
    for discipline in ['PLAT','ATTELE_AUTOSTART','ATTELE_VOLTE','TROT_MONTE','HAIES','STEEPLECHASE','CROSS_COUNTRY']:
        out['discipline_'+discipline] = q.discipline.eq(discipline).astype(float)
    for sex in ['FEMELLES','HONGRES']:
        out['sex_'+sex] = q.sex.fillna('').eq(sex).astype(float)
    # Interactions physiques justifiées; pas de bonus arbitraire de corde.
    out['plat_draw'] = out.draw*out.discipline_PLAT
    out['plat_weight'] = out.weight*out.discipline_PLAT
    out['autostart_draw'] = out.draw*out.discipline_ATTELE_AUTOSTART
    h['top5'] = pd.to_numeric(h.finish_position,errors='coerce').between(1,5).astype(float)
    h['win'] = pd.to_numeric(h.finish_position,errors='coerce').eq(1).astype(float)
    for col in ['horse_name','jockey','trainer']:
        hk, qk = key(h[col]), key(q[col])
        events = h.assign(entity=hk, starts=1)
        events = events[events.entity.ne('')]
        daily = events.groupby(['entity','race_date'])[['starts','top5','win']].sum().sort_index()
        daily = daily.groupby(level=0).cumsum().reset_index()
        left = pd.DataFrame({'entity':qk,'race_date':q.race_date,'idx':q.index}).sort_values('race_date')
        if daily.empty:
            merged = left.assign(starts=0.,top5=0.,win=0.)
        else:
            merged = pd.merge_asof(left,daily.sort_values('race_date'),on='race_date',by='entity',allow_exact_matches=False).sort_values('idx')
        merged = merged.set_index('idx').reindex(q.index)
        starts = merged.starts.fillna(0)
        out[col+'_starts'] = np.log1p(starts)
        out[col+'_top5'] = (merged.top5.fillna(0)+5)/(starts+10)
        out[col+'_win'] = (merged.win.fillna(0)+1)/(starts+10)
    # Forme calculée sur les cinq dernières courses validées, sans musique rétrospective.
    h['entity'] = key(h.horse_name)
    h['performance'] = np.where(h.finish_position.notna(),(1-(h.finish_position-1)/(h.groupby('race_id').horse_number.transform('size').clip(lower=2)-1)).clip(0,1),0.)
    h = h.sort_values(['race_date','race_id','horse_number'])
    h['form'] = h.groupby('entity').performance.transform(lambda s:s.rolling(5,min_periods=1).mean())
    h['consistency'] = h.groupby('entity').performance.transform(lambda s:1-s.rolling(5,min_periods=2).std(ddof=0))
    daily = h.drop_duplicates(['entity','race_date'],keep='last')[['entity','race_date','form','consistency']]
    left = pd.DataFrame({'entity':key(q.horse_name),'race_date':q.race_date,'idx':q.index}).sort_values('race_date')
    if daily.empty:
        merged=left.assign(form=np.nan,consistency=np.nan)
    else:
        merged=pd.merge_asof(left,daily.sort_values('race_date'),on='race_date',by='entity',allow_exact_matches=False)
    merged=merged.set_index('idx').reindex(q.index)
    out['form'] = merged.form.fillna(.45)
    out['consistency'] = merged.consistency.fillna(.5)
    return out.astype(float)


def predict_scores(artifact, features):
    x = features[artifact['features']].to_numpy()
    z = ((x-np.array(artifact['mean']))/np.array(artifact['scale'])) @ np.array(artifact['coef']) + artifact['intercept']
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--history',required=True,help='CSV historique validé produit par run_analysis.py')
    p.add_argument('--race',required=True,help='CSV avec partants et variables pré-course; jamais de résultats nécessaires')
    p.add_argument('--model',default='candidate.json')
    p.add_argument('--output',default='pronostic.csv')
    args=p.parse_args()
    h=pd.read_csv(args.history); q=pd.read_csv(args.race)
    q=q[~q.get('is_non_runner',pd.Series(False,index=q.index)).fillna(False).astype(bool)].copy()
    if q.duplicated(['race_id','horse_number']).any(): raise ValueError('Doublon de partant')
    if pd.to_datetime(q.race_date,errors='coerce').isna().any(): raise ValueError('Date manquante')
    a=json.loads(Path(args.model).read_text())
    x=prepare(h,q); q['score_top5']=predict_scores(a,x)
    q=q.sort_values(['race_id','score_top5','horse_number'],ascending=[True,False,True])
    q['rank']=q.groupby('race_id').cumcount()+1
    q.to_csv(args.output,index=False)
    print(q[['race_id','horse_number','horse_name','rank','score_top5']].to_string(index=False))

if __name__=='__main__': main()
