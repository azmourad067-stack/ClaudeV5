# Minimal parser test without importing Streamlit app (Streamlit absent in build container).
from quinte_v45 import apply_v45_market_dynamics
import pandas as pd

base=pd.DataFrame({
 'discipline':['PLAT']*3,
 'horse_number':[1,2,3],
 'selected_top7':[True]*3,
 'quinte_rank':[1,2,3],
 'v44_winner_score':[0.8,0.75,0.7],
 'top5_probability':[0.4,0.35,0.3],
 'odds':[4.0,6.0,8.0],
 'odds_direct':[4.0,6.0,8.0],
 'odds_reference':[5.0,5.5,8.0],
})
out=apply_v45_market_dynamics(base,{},minutes_to_start=30)
assert len(out)==3
assert out.selected_top7.all()
assert out.v45_dynamics_confidence.between(0,1).all()
assert out.v45_projected_odds_t2.notna().all()
print('OK parser/dynamics fallback direct-reference')
print(out[['horse_number','v45_projected_odds_t2','v45_projected_change_pct','v45_dynamics_score']].to_string(index=False))
