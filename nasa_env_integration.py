"""
Replace the SIMULATED cabin environment with REAL ISS cabin telemetry from the NASA OSDR Environmental Data App (EDA).

Input 1: EDA table (CSV/TSV) with columns: Controller_Time_GMT, Temp_degC_ISS, RH_percent_ISS, CO2_ppm_ISS (+ ground columns)
Input 2: window_features_context.csv from astro_mission_pipeline.py
Output : window_features_real_env.csv  (temp_c, rh_pct, co2_ppm = real ISS values; env_load_index recomputed)

The real record (e.g. mission RR-1, ~5-minute sampling) is looped over the simulated mission timeline, so the
crew activity is still simulated but the cabin conditions are real NASA measurements. Say so in your write-up.

Usage:
  python nasa_env_integration.py --eda data/nasa/eda_rr1.csv --features out/window_features_context.csv --out out/
  python astro_health_module.py --features out/window_features_real_env.csv --inject --out out/
"""
import argparse, os
import numpy as np, pandas as pd

COLS = {"time": "Controller_Time_GMT", "temp": "Temp_degC_ISS", "rh": "RH_percent_ISS", "co2": "CO2_ppm_ISS"}

def load_eda(path):
    d = pd.read_csv(path, sep=None, engine="python")
    d.columns = [c.strip() for c in d.columns]
    miss = [v for v in COLS.values() if v not in d.columns]
    if miss: raise SystemExit(f"Missing columns {miss}. Found: {list(d.columns)}")
    d["time"] = pd.to_datetime(d[COLS["time"]], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    for k in ("temp", "rh", "co2"): d[k] = pd.to_numeric(d[COLS[k]], errors="coerce")
    d = d.dropna(subset=["temp", "rh", "co2"])
    d["t"] = (d.time - d.time.iloc[0]).dt.total_seconds()
    return d[["time", "t", "temp", "rh", "co2"]].reset_index(drop=True)

def apply_real_env(w, eda):
    w = w.copy()
    dur = float(eda.t.iloc[-1])
    if dur <= 0: raise SystemExit("EDA record needs at least 2 time points.")
    env_t = w.t_s.to_numpy() % dur                                  # loop the real record over the mission
    for k, col in (("temp", "temp_c"), ("rh", "rh_pct"), ("co2", "co2_ppm")):
        w[col] = np.interp(env_t, eda.t.to_numpy(), eda[k].to_numpy())
    # environment load = deviation from the record's own typical conditions (scale-free, 30-min rolling mean)
    med = eda[["temp", "rh", "co2"]].median()
    dev = (w.temp_c - med.temp).abs()/1.0 + (w.rh_pct - med.rh).abs()/5.0 + (w.co2_ppm - med.co2).abs()/500.0
    r = int(30 * 60 / 30)                                           # 30-min window at 30 s per row
    w["env_load_index"] = (dev / 3).rolling(r, min_periods=1).mean()
    w["env_source"] = "NASA OSDR EDA (ISS cabin telemetry, looped)"
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    eda = load_eda(a.eda)
    print(f"EDA rows: {len(eda)} | {eda.time.iloc[0]} -> {eda.time.iloc[-1]}")
    print(eda[["temp", "rh", "co2"]].describe().loc[["mean", "min", "max"]].round(1).to_string())
    w = apply_real_env(pd.read_csv(a.features), eda)
    os.makedirs(a.out, exist_ok=True)
    w.to_csv(f"{a.out}/window_features_real_env.csv", index=False)
    print(f"Saved {a.out}/window_features_real_env.csv")

if __name__ == "__main__":
    main()
