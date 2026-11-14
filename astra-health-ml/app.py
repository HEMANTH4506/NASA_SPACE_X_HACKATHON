"""Optional Streamlit UI (not required).  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import config as C
from predict import OUT_COLS, load_models, predict_dataframe

st.set_page_config(page_title="ASTRA-HEALTH", layout="wide")
st.title("ASTRA-HEALTH · crew risk prediction")
st.caption("Experimental decision-support prototype. Not diagnostic.")


@st.cache_resource
def _models():
    return load_models()


try:
    models = _models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

up = st.file_uploader("Upload mission CSV (same columns as the training data)", type="csv")
raw = pd.read_csv(up) if up else pd.read_csv(C.DATA_PATH)
st.write(f"Using **{'uploaded file' if up else 'bundled dataset'}** - {len(raw)} rows")

try:
    out = predict_dataframe(raw, models)
except ValueError as e:
    st.error(f"Input problem: {e}")
    st.stop()

crew = st.selectbox("Astronaut", sorted(out.crew_id.unique()))
sub = out[out.crew_id == crew].sort_values("mission_day")
day = st.slider("Mission day", int(sub.mission_day.min()), int(sub.mission_day.max()), int(sub.mission_day.max()))
row = sub[sub.mission_day == day].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted risk", row.pred_risk)
c2.metric("Predicted readiness", f"{row.pred_readiness_pct:.1f}%")
if "p_atrisk_tomorrow" in row:
    c3.metric("P(at risk tomorrow)", f"{row.p_atrisk_tomorrow:.0%}",
              "early warning" if row.early_warning else "no alert", delta_color="inverse")

st.subheader("Trend")
st.line_chart(sub.set_index("mission_day")[["pred_readiness_pct"]])
st.line_chart(sub.set_index("mission_day")[["p_warning", "p_critical"]])
st.subheader("Predictions")
st.dataframe(sub[[c for c in OUT_COLS if c in sub.columns]], use_container_width=True)
st.download_button("Download all predictions (CSV)",
                   out[[c for c in OUT_COLS if c in out.columns]].to_csv(index=False), "predictions.csv")
