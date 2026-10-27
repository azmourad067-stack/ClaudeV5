"""Exécute streamlit_app.py avec un faux module streamlit et des données PMU simulées
(un Quinté forward réel), pour vérifier que tout le chemin d'analyse s'exécute sans erreur."""
from __future__ import annotations
import sys, types, contextlib
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from common import load_history, load_forward

LOG = []


class Any_:
    def __init__(self, name="st"): self._n = name
    def __getattr__(self, k): return Any_(f"{self._n}.{k}")
    def __call__(self, *a, **k):
        LOG.append((self._n, a[:1]))
        if self._n.endswith("error") or self._n.endswith("exception"):
            raise SystemExit(f"APP ERROR: {a}")
        return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([Any_(f"{self._n}[{i}]") for i in range(4)])


st = types.ModuleType("streamlit")
for name in ["title", "caption", "metric", "success", "info", "warning", "markdown", "subheader", "dataframe",
             "json", "download_button", "spinner", "expander", "set_page_config", "stop"]:
    setattr(st, name, Any_(name))
st.error = Any_("error"); st.exception = Any_("exception")
st.columns = lambda n, **k: [Any_(f"col{i}") for i in range(n if isinstance(n, int) else len(n))]
st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st.cache_resource = st.cache_data
for f in ["clear"]:
    pass
st.secrets = {}
st.session_state = {}
st.date_input = lambda *a, **k: date(2026, 10, 22)
st.button = lambda label, **k: "Analyser" in label
st.checkbox = lambda *a, **k: COMPARE
st.selectbox = lambda label, opts, **k: opts[0]
sys.modules["streamlit"] = st

COMPARE = len(sys.argv) > 1 and sys.argv[1] == "compare"
src = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
head, tail = src.split("# INTERFACE", 1)
g = {"__name__": "app", "__file__": str(ROOT / "streamlit_app.py")}
exec(compile(head, "streamlit_app.py", "exec"), g)

h = load_history(); fwd, arrivals = load_forward("snap")
rid = "R1C1_2026-10-22"
race = fwd[fwd.race_id == rid].drop(columns=["finish_position"]).assign(
    finish_position=np.nan, odds_direct=lambda d: d.odds, odds_reference=np.nan, trend_indicator="",
    trend_strength=np.nan).reset_index(drop=True)


class _Clearable:
    def __init__(self, f): self.f = f
    def __call__(self, *a, **k): return self.f(*a, **k)
    def clear(self): pass


g["discover_daily_program"] = _Clearable(lambda d: pd.DataFrame([{
    "meeting_number": 1, "race_number": 1, "hippodrome": race.hippodrome.iloc[0], "label": "Quinté test",
    "start_time": "13:55", "distance_m": race.distance.iloc[0], "field_size_pmu": len(race), "is_quinte": True}]))
g["get_programme"] = g["get_course_detail"] = g["get_participants"] = _Clearable(lambda *a, **k: {})
g["participants_to_frame"] = lambda *a, **k: race.copy()
g["build_history"] = lambda d: (h[h.race_date < pd.Timestamp(d)].copy(), "test")
g["_minutes_to_start"] = lambda s: 20.0
g["save_market_snapshot"] = lambda *a, **k: (0, "test")
g["load_saved_market_history"] = lambda *a, **k: {}
exec(compile("# INTERFACE" + tail, "streamlit_app.py", "exec"), g)
shown = [x for x in LOG if x[0] in ("markdown", "success", "info", "warning")]
for s in shown:
    print(s[0], str(s[1][0])[:160] if s[1] else "")
print("arrivée réelle :", arrivals[rid])
print("HARNESS OK")
