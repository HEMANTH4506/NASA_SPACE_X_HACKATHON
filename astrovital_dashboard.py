"""
AstroVital dashboard (Streamlit). Run:  streamlit run astrovital_dashboard.py

Needs (in the same folder): astro_health_module.py and a window-features CSV from the pipeline, e.g.
  out/window_features_real_env.csv   (real NASA ISS cabin data)  or  out/window_features_context.csv (simulated)
pip install streamlit altair pandas numpy scikit-learn
"""
import os, sys
import numpy as np, pandas as pd, altair as alt, streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from astro_health_module import inject_hr_anomaly, hourly_features, analyse, evaluate

st.set_page_config(page_title="AstroVital", page_icon="🛰️", layout="wide")
COLORS = {"GREEN": "#2e9e5b", "AMBER": "#f0a030", "RED": "#d64545"}
CANDIDATES = ["out/window_features_real_env.csv", "out/window_features_context.csv"]

@st.cache_data
def load_windows(path):
    return pd.read_csv(path)

@st.cache_data
def run_analysis(path, baseline_hours, inject, start_h, hours, delta):
    w = load_windows(path)
    injected = set()
    if inject:
        w, injected = inject_hr_anomaly(w, start_h, hours, delta)
    h = hourly_features(w)
    if "co2_ppm" in w:
        co2 = w.assign(hour=(w.t_s // 3600).astype(int)).groupby("hour").co2_ppm.mean()
        h["co2_ppm"] = co2.reindex(h.index).ffill().bfill()
    h = analyse(h, baseline_hours)
    h.index.name = "hour"
    real_env = "env_source" in w.columns
    return h, injected, real_env

# ---------------- sidebar ----------------
st.sidebar.title("🛰️ AstroVital")
paths = [p for p in CANDIDATES if os.path.exists(os.path.join(HERE, p))]
if not paths:
    st.error("No window-features CSV found. Run:  python astro_mission_pipeline.py --days 5 --out out/")
    st.stop()
path = os.path.join(HERE, st.sidebar.selectbox("Data file", paths))
baseline_hours = st.sidebar.slider("Baseline period (hours)", 48, 72, 48, 12)
st.sidebar.subheader("Anomaly test")
inject = st.sidebar.checkbox("Simulate heart-rate anomaly", value=True)
start_h = st.sidebar.slider("Anomaly start hour", 50, 110, 72)
hours = st.sidebar.slider("Duration (hours)", 1, 12, 6)
delta = st.sidebar.slider("Resting HR increase (bpm)", 5, 25, 12)
co2_limit = st.sidebar.slider("CO2 attention level (ppm)", 2000, 6000, 4000, 250,
                              help="Adjustable demo threshold, not an official limit.")

h, injected, real_env = run_analysis(path, baseline_hours, inject, start_h, hours, float(delta))
max_hour = int(h.index.max())
hour = st.sidebar.slider("Mission hour (replay)", baseline_hours, max_hour, max_hour)
view = h.loc[:hour].reset_index()
cur = h.loc[hour]

# ---------------- header ----------------
st.title("AstroVital: Astronaut Health Monitoring")
if real_env:
    st.caption("Cabin environment: **REAL** NASA ISS telemetry (OSDR Environmental Data App). "
               "Crew activity and heart rate: **SIMULATED**.")
else:
    st.caption("All data in this view is **SIMULATED**.")
tab_crew, tab_ground, tab_about = st.tabs(["Crew view", "Ground team view", "About / method"])

# ---------------- crew view ----------------
with tab_crew:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"<h1 style='color:{COLORS[cur.alert]};margin:0'>{cur.alert}</h1>", unsafe_allow_html=True)
        if pd.notna(cur.readiness):
            st.metric("Readiness (0-100)", f"{cur.readiness:.0f}")
        else:
            st.info("Collecting 24 h of history for readiness.")
        st.write(cur.reason if cur.reason else "Within your personal baseline.")
    with c2:
        st.subheader("What drives readiness")
        for label, col in [("Sleep", "sleep_score"), ("Heart rate vs baseline", "hr_score"),
                           ("Exercise load", "load_score"), ("Cabin environment", "env_score")]:
            v = cur[col]
            st.write(f"{label}: {'n/a' if pd.isna(v) else f'{v:.0f}/100'}")
            st.progress(0.0 if pd.isna(v) else float(np.clip(v / 100, 0, 1)))
    if cur.get("co2_ppm", 0) > co2_limit:
        st.warning(f"Cabin CO2 is {cur.co2_ppm:.0f} ppm, above the {co2_limit} ppm attention level.")

# ---------------- ground view ----------------
with tab_ground:
    sel = alt.selection_interval(bind="scales", encodings=["x"])
    view["expected_hr"] = view.rest_hr - view.hr_resid
    base = alt.Chart(view).encode(x=alt.X("hour:Q", title="Mission hour"))
    hr_chart = (base.mark_line(color="#888").encode(y=alt.Y("rest_hr:Q", title="Resting-state HR (bpm)", scale=alt.Scale(zero=False)))
                + base.mark_line(strokeDash=[4, 4], color="#4a90d9").encode(y="expected_hr:Q")
                + base.mark_circle(size=60).encode(y="rest_hr:Q", tooltip=["hour", "rest_hr", "alert", "reason"],
                    color=alt.Color("alert:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=alt.Legend(title="Alert"))))
    st.subheader("Resting heart rate vs personal baseline (dashed)")
    st.altair_chart(hr_chart.add_params(sel).properties(height=260), use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Readiness")
        st.altair_chart(alt.Chart(view.dropna(subset=["readiness"])).mark_line(color="#4a90d9").encode(
            x="hour:Q", y=alt.Y("readiness:Q", scale=alt.Scale(domain=[0, 100]))).properties(height=220), use_container_width=True)
    with b:
        st.subheader("Exercise and sleep (min per hour)")
        m = view.melt("hour", ["exercise_min", "sleep_min"], "type", "minutes")
        st.altair_chart(alt.Chart(m).mark_bar().encode(x="hour:Q", y="minutes:Q", color="type:N").properties(height=220), use_container_width=True)

    st.subheader("Cabin environment" + (" (real NASA ISS data)" if real_env else " (simulated)"))
    e1, e2, e3 = st.columns(3)
    for col, key, title in [(e1, "temp_c", "Temperature (°C)"), (e2, "rh_pct", "Humidity (%)"), (e3, "co2_ppm", "CO2 (ppm)")]:
        if key in view:
            ch = alt.Chart(view).mark_line().encode(x="hour:Q", y=alt.Y(f"{key}:Q", title=title, scale=alt.Scale(zero=False))).properties(height=180)
            if key == "co2_ppm":
                ch = ch + alt.Chart(pd.DataFrame({"y": [co2_limit]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
            col.altair_chart(ch, use_container_width=True)

    st.subheader("Alert log")
    log = view[view.alert != "GREEN"][["hour", "alert", "rest_hr", "hr_resid", "readiness", "reason"]].round(1)
    st.dataframe(log, use_container_width=True, hide_index=True)
    if injected:
        p, r, d = evaluate(h, injected)
        st.caption(f"Test anomaly (hours {min(injected)}-{max(injected)}): precision {p:.2f}, recall {r:.2f}, "
                   f"detection delay {d} h. Simulated anomaly, not real-world accuracy.")

# ---------------- about ----------------
with tab_about:
    st.markdown("""
**Pipeline:** wearable and cabin data -> edge processing (30 s windows) -> activity recognition and intensity ->
personal baseline (first 48 h) -> anomaly detection (EWMA z-score + Isolation Forest) -> readiness score -> alerts.

**Data:** crew activity and heart rate are simulated; activity models validated on PAMAP2 (Reiss & Stricker, DFKI/UCI).
Cabin temperature, humidity and CO2 are real ISS telemetry from NASA OSDR's Environmental Data App (mission RR-1, 2014).

**Limits:** readiness weights and alert thresholds are tunable starting values, not clinically validated.
AstroVital supports decisions and does not diagnose; it is not a medical device.
""")
