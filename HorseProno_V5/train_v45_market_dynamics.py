"""Entraînement futur du coefficient de mouvement de cote T-30 -> T-2.

Entrée : export CSV de public.quinte_market_snapshots.
Aucune dépendance sklearn : numpy/pandas uniquement.

Le script prend, pour chaque cheval/course :
- un snapshot T-30 (15 à 40 minutes avant le départ),
- un snapshot T-2 (0 à 5 minutes avant le départ),
et apprend une régression ridge légère du log-mouvement de cote.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=["log_direct_reference_gap","log_odds","market_probability","minutes_scaled"]

def _select_nearest(g: pd.DataFrame, target: float, lo: float, hi: float):
    z=g[(g.minutes_to_start>=lo)&(g.minutes_to_start<=hi)].copy()
    if z.empty: return None
    return z.iloc[(z.minutes_to_start-target).abs().argmin()]

def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    for c in ["minutes_to_start","odds","odds_direct","odds_reference","market_probability"]:
        df[c]=pd.to_numeric(df.get(c),errors="coerce")
    rows=[]
    keys=["race_date","meeting_number","race_number","horse_number"]
    for key,g in df.groupby(keys,dropna=False):
        a=_select_nearest(g,30,15,40); b=_select_nearest(g,2,0,5)
        if a is None or b is None: continue
        o0=float(a.odds_direct if pd.notna(a.odds_direct) else a.odds)
        o1=float(b.odds_direct if pd.notna(b.odds_direct) else b.odds)
        if not (o0>1 and o1>1): continue
        ref=float(a.odds_reference) if pd.notna(a.odds_reference) and float(a.odds_reference)>1 else o0
        rows.append({
            **dict(zip(keys,key)),
            "y":np.log(o1/o0),
            "log_direct_reference_gap":np.log(o0/ref),
            "log_odds":np.log(o0),
            "market_probability":float(a.market_probability) if pd.notna(a.market_probability) else 0.0,
            "minutes_scaled":float(a.minutes_to_start)/30.0,
        })
    return pd.DataFrame(rows)

def fit_ridge(X,y,lam=1.0):
    X=np.asarray(X,float); y=np.asarray(y,float)
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1
    Z=(X-mu)/sd
    A=np.c_[np.ones(len(Z)),Z]
    reg=np.eye(A.shape[1])*lam; reg[0,0]=0
    coef=np.linalg.solve(A.T@A+reg,A.T@y)
    pred=A@coef
    return coef,mu,sd,float(np.sqrt(np.mean((y-pred)**2))),float(np.mean(np.abs(y-pred)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out",default="v45_market_artifact.json")
    args=ap.parse_args()
    df=pd.read_csv(args.csv,low_memory=False)
    pairs=build_pairs(df)
    if len(pairs)<50:
        raise SystemExit(f"Pas assez de paires T30/T2 : {len(pairs)} (minimum conseillé : 50).")
    pairs["race_date"]=pd.to_datetime(pairs.race_date)
    dates=sorted(pairs.race_date.dropna().unique())
    cut=dates[max(1,int(len(dates)*0.75))-1]
    train=pairs[pairs.race_date<=cut]; test=pairs[pairs.race_date>cut]
    coef,mu,sd,rmse,mae=fit_ridge(train[FEATURES],train.y,lam=1.0)
    art={
      "model_name":"HorseProno_V4_5_market_move_ridge",
      "target":"log(odds_T2 / odds_T30)",
      "features":FEATURES,
      "intercept":float(coef[0]),
      "coefficients":{f:float(v) for f,v in zip(FEATURES,coef[1:])},
      "mean":{f:float(v) for f,v in zip(FEATURES,mu)},
      "std":{f:float(v) for f,v in zip(FEATURES,sd)},
      "train_rows":len(train),"test_rows":len(test),"train_end":str(pd.Timestamp(cut).date()),
      "train_rmse":rmse,"train_mae":mae,
    }
    if len(test):
      Z=(test[FEATURES].to_numpy(float)-mu)/sd; pred=coef[0]+Z@coef[1:]
      art["test_rmse"]=float(np.sqrt(np.mean((test.y.to_numpy(float)-pred)**2)))
      art["test_mae"]=float(np.mean(np.abs(test.y.to_numpy(float)-pred)))
    Path(args.out).write_text(json.dumps(art,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(art,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
