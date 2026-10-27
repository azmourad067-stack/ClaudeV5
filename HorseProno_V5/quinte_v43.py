"""HorseProno Quinté V4.3 — Outsider Detector (shadow mode).

La V4.3 conserve intégralement le Top7 V4.2 en production. Sur les courses de PLAT,
elle évalue en parallèle les chevaux absents à la fois des Top7 V2 et V3. Le meilleur
outsider est affiché avec une probabilité de détection et le cheval V4.2 qu'il aurait
remplacé. Le seuil de recherche est 0.50, mais aucun remplacement n'est appliqué dans
cette version shadow : les prédictions forward doivent d'abord confirmer le signal.
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from quinte_v42 import rank_quinte_v42


def load_outsider_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sigmoid(z: float) -> float:
    z=max(-40.0,min(40.0,float(z)))
    return 1.0/(1.0+math.exp(-z))


def _recent_features(value: Any) -> dict[str,float]:
    vals=[10 if int(x)==0 else min(int(x),20) for x in re.findall(r"(?<!\d)(\d{1,2})p", str(value or ""), flags=re.I)][:8]
    if not vals:
        return {k:0.0 for k in ["rf_n","rf_last","rf_mean3","rf_mean5","rf_top3_5","rf_top5_5","rf_win5","rf_bad5","rf_trend"]}
    q=[max(0.0,min(1.0,(11-v)/10.0)) for v in vals]
    mean=lambda x: float(np.mean(x)) if len(x) else 0.0
    return {
        "rf_n":float(len(vals)), "rf_last":q[0], "rf_mean3":mean(q[:3]), "rf_mean5":mean(q[:5]),
        "rf_top3_5":mean([v<=3 for v in vals[:5]]), "rf_top5_5":mean([v<=5 for v in vals[:5]]),
        "rf_win5":mean([v==1 for v in vals[:5]]), "rf_bad5":mean([v>=10 for v in vals[:5]]),
        "rf_trend":mean(q[:3])-mean(q[3:6] if len(q)>=4 else q[:3]),
    }


def _feature_frame(result: pd.DataFrame) -> pd.DataFrame:
    out=result.copy()
    n=max(len(out),1)
    draw=pd.to_numeric(out.get("draw"), errors="coerce")
    weight=pd.to_numeric(out.get("weight"), errors="coerce")
    out["draw_rel"]=(draw-1.0)/max(n-1,1); out["draw_rel"]=out["draw_rel"].fillna(0.5)
    if weight.notna().sum()>=2 and float(weight.max())>float(weight.min()):
        out["weight_rel"]=(weight-weight.min())/(weight.max()-weight.min())
    else: out["weight_rel"]=0.5
    sd=float(weight.std(ddof=0)) if weight.notna().sum()>=2 else 0.0
    out["weight_z"]=(weight-weight.mean())/(sd if sd>0 else 1.0); out["weight_z"]=out["weight_z"].fillna(0.0)
    distance=pd.to_numeric(out.get("distance"), errors="coerce").fillna(1800.0)
    age=pd.to_numeric(out.get("age"), errors="coerce").fillna(5.0)
    odds=pd.to_numeric(out.get("odds"), errors="coerce")
    out["market_pct"]=pd.to_numeric(out["market_rankpct"],errors="coerce").fillna(0.0)
    out["logit_pct"]=pd.to_numeric(out["model_rankpct"],errors="coerce").fillna(0.0)
    out["ranker_pct"]=pd.to_numeric(out["ranker_percentile"],errors="coerce").fillna(0.0)
    out["market_rank_norm"]=pd.to_numeric(out["market_rank"],errors="coerce").fillna(n)/float(n)
    out["log_odds"]=np.log1p(odds.where(odds>0)).fillna(4.0)
    out["field_size"]=float(n)/20.0; out["distance"]=distance/3000.0
    out["sprint"]=(distance<=1400).astype(int); out["mid"]=((distance>=1500)&(distance<=2000)).astype(int); out["long"]=(distance>=2100).astype(int)
    out["outer_draw"]=(out.draw_rel>=2/3).astype(int); out["inner_draw"]=(out.draw_rel<=1/3).astype(int)
    out["light_weight"]=(out.weight_rel<=1/3).astype(int); out["heavy_weight"]=(out.weight_rel>=2/3).astype(int)
    out["draw_sprint"]=out.draw_rel*out.sprint; out["draw_long"]=out.draw_rel*out.long
    out["light_sprint"]=out.light_weight*out.sprint; out["light_long"]=out.light_weight*out.long
    out["age"]=age/10.0
    recent=pd.DataFrame([_recent_features(v) for v in out.get("recent_form", pd.Series([""]*len(out)))],index=out.index)
    for c in recent.columns: out[c]=recent[c]
    # Noms identiques au jeu d'apprentissage OOF.
    out["horse_top5"]=pd.to_numeric(out["horse_name_top5"],errors="coerce").fillna(0.0)
    out["horse_win"]=pd.to_numeric(out["horse_name_win"],errors="coerce").fillna(0.0)
    return out


def rank_quinte_v43(history: pd.DataFrame, race: pd.DataFrame, artifact: dict[str,Any], ranker: Any, outsider_artifact: dict[str,Any]) -> pd.DataFrame:
    result=rank_quinte_v42(history,race,artifact,ranker).copy()
    result["v43_mode"]="V42_BASE"
    result["v43_outsider_probability"]=np.nan
    result["v43_outsider_candidate"]=False
    result["v43_shadow_threshold_met"]=False
    result["v43_shadow_replacement"]=False
    result["v43_shadow_replacement_number"]=np.nan
    if str(result.iloc[0].get("discipline","")).strip().upper()!="PLAT":
        return result
    x=_feature_frame(result)
    outside=x.loc[~(x["in_v2_top7"].astype(bool)|x["in_v3_top7"].astype(bool))].copy()
    if outside.empty: return result
    co=outsider_artifact["coefficients"]; features=outsider_artifact["features"]
    for c in features:
        if c not in outside.columns: outside[c]=0.0
    z=np.full(len(outside),float(outsider_artifact["intercept"]),dtype=float)
    for c in features: z += float(co.get(c,0.0))*pd.to_numeric(outside[c],errors="coerce").fillna(0.0).to_numpy(dtype=float)
    outside["_p"]=[_sigmoid(v) for v in z]
    best=outside.sort_values(["_p","top5_probability","horse_number"],ascending=[False,False,True]).iloc[0]
    candidate=int(best.horse_number); p=float(best._p); threshold=float(outsider_artifact.get("threshold",0.50))
    # Le remplacement théorique ne touche jamais le consensus V2/V3.
    selected=result[result.selected_top7.astype(bool)].copy()
    repl=selected.loc[~selected["v2_v3_consensus"].astype(bool)].sort_values(["v42_edge_score","top5_probability","horse_number"],ascending=[True,True,False])
    repl_num=int(repl.iloc[0].horse_number) if not repl.empty else None
    idx=result.index[result.horse_number.astype(int).eq(candidate)]
    if len(idx):
        result.loc[idx,"v43_outsider_probability"]=p
        result.loc[idx,"v43_outsider_candidate"]=True
        result.loc[idx,"v43_shadow_threshold_met"]=bool(p>=threshold)
    if repl_num is not None:
        result["v43_shadow_replacement_number"]=float(repl_num)
        ridx=result.index[result.horse_number.astype(int).eq(repl_num)]
        if len(ridx): result.loc[ridx,"v43_shadow_replacement"]=True
    result["v43_mode"]="OUTSIDER_SHADOW"
    return result
