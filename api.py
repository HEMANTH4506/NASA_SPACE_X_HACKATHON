"""
AstroVital API (FastAPI): serves the health analysis as JSON for any front end (v0, Lovable, mobile, mission systems).

Run:   pip install fastapi uvicorn pandas numpy scikit-learn
       uvicorn api:app --reload            (then open http://127.0.0.1:8000/docs for the interactive docs)

Endpoints
  GET /                      service info and data source
  GET /timeline              hourly records (readiness, alert, reason, environment ...)
  GET /alerts                only AMBER / RED hours
  GET /status?hour=75        one hour (default: latest)
  GET /simulate-anomaly      re-run with an injected resting-HR anomaly, returns timeline + detection metrics
Common query params: baseline (hours, default 48), inject (bool), start, hours, delta
Alerts are decision support, not a medical diagnosis.
"""
import os, sys
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from astro_health_module import inject_hr_anomaly, hourly_features, analyse, evaluate

CANDIDATES = ["out/window_features_real_env.csv", "out/window_features_context.csv"]
FIELDS = ["hour", "activity", "rest_hr", "hr_resid", "hr_z_ewma", "readiness", "sleep_score", "hr_score",
          "load_score", "env_score", "temp_c", "rh_pct", "co2_ppm", "env_load", "exercise_min", "sleep_min",
          "alert", "reason"]

app = FastAPI(title="AstroVital API", version="0.1",
              description="Astronaut health monitoring analytics. Crew data simulated; cabin environment real NASA ISS data when available.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _data_path():
    for p in CANDIDATES:
        if os.path.exists(os.path.join(HERE, p)):
            return os.path.join(HERE, p)
    raise HTTPException(503, "No window-features CSV found. Run astro_mission_pipeline.py (and nasa_env_integration.py) first.")

def analyse_file(baseline=48, inject=False, start=72, hours=6, delta=12.0):
    path = _data_path()
    w = pd.read_csv(path)
    injected = set()
    if inject:
        w, injected = inject_hr_anomaly(w, start, hours, delta)
    h = hourly_features(w)
    hr = (w.t_s // 3600).astype(int)
    if "co2_ppm" in w:
        h["co2_ppm"] = w.groupby(hr).co2_ppm.mean().reindex(h.index).ffill().bfill()
    if "act_pred" in w:
        h["activity"] = w.groupby(hr).act_pred.agg(lambda s: s.mode().iloc[0]).reindex(h.index)
    h = analyse(h, baseline)
    h.index.name = "hour"
    return h.reset_index(), injected, ("env_source" in w.columns), os.path.relpath(path, HERE)

def to_records(df):
    cols = [c for c in FIELDS if c in df.columns]
    d = df[cols].round(2).astype(object)
    d = d.where(d.notna(), None)
    return d.to_dict(orient="records")

@app.get("/")
def info():
    h, _, real_env, src = analyse_file()
    return {"service": "AstroVital API", "data_file": src, "hours": len(h),
            "environment": "real NASA ISS telemetry (OSDR EDA)" if real_env else "simulated",
            "crew_data": "simulated", "docs": "/docs"}

@app.get("/timeline")
def timeline(baseline: int = Query(48, ge=48, le=72)):
    h, _, _, _ = analyse_file(baseline)
    return to_records(h)

@app.get("/alerts")
def alerts(baseline: int = Query(48, ge=48, le=72)):
    h, _, _, _ = analyse_file(baseline)
    return to_records(h[h.alert != "GREEN"])

@app.get("/status")
def status(hour: int | None = None, baseline: int = Query(48, ge=48, le=72)):
    h, _, _, _ = analyse_file(baseline)
    hour = int(h.hour.max()) if hour is None else hour
    row = h[h.hour == hour]
    if row.empty:
        raise HTTPException(404, f"hour {hour} not in range 0-{int(h.hour.max())}")
    return to_records(row)[0]

@app.get("/simulate-anomaly")
def simulate(start: int = Query(72, ge=50, le=110), hours: int = Query(6, ge=1, le=12),
             delta: float = Query(12.0, ge=5, le=25), baseline: int = Query(48, ge=48, le=72)):
    h, injected, _, _ = analyse_file(baseline, True, start, hours, delta)
    p, r, d = evaluate(h.set_index("hour"), injected)
    num = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)
    return {"injected_hours": sorted(injected), "precision": num(p), "recall": num(r),
            "detection_delay_hours": d, "note": "Simulated anomaly, not real-world accuracy.",
            "timeline": to_records(h)}
