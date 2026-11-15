import argparse, os
import numpy as np, pandas as pd

WIN_MIN = 0.5                                   # window length in minutes (30 s)
EXERCISE = {"treadmill", "cycling", "resistive"}
W_SLEEP, W_HR, W_LOAD, W_ENV = 0.35, 0.30, 0.20, 0.15

def inject_hr_anomaly(w, start_h, hours, delta):
    """Test helper: raise heart rate by `delta` bpm for `hours` hours starting at hour `start_h`."""
    w = w.copy()
    hr = (w.t_s // 3600)
    m = (hr >= start_h) & (hr < start_h + hours)
    w.loc[m, "hr_mean"] += delta
    return w, set(range(start_h, start_h + hours))

def hourly_features(w):
    w = w.copy()
    w["hour"] = (w.t_s // 3600).astype(int)
    motion_int = "int_pred" if "int_pred" in w else "intensity"           # motion-only intensity if available
    w["sed_hr"] = w.hr_mean.where(w[motion_int] == "sedentary")
    w["ex"] = w.act_pred.isin(EXERCISE).astype(float)
    w["sl"] = (w.act_pred == "sleep").astype(float)
    w["zone"] = ((w.hrr >= 0.4) & (w.ex > 0)).astype(float)
    h = w.groupby("hour").agg(rest_hr=("sed_hr", "median"), motion=("acc_mad", "mean"),
                              env_load=("env_load_index", "mean"), temp_c=("temp_c", "mean"),
                              rh_pct=("rh_pct", "mean"), ex=("ex", "sum"), sl=("sl", "sum"), zone=("zone", "sum"))
    h["exercise_min"], h["sleep_min"], h["hr_zone_min"] = h.ex * WIN_MIN, h.sl * WIN_MIN, h.zone * WIN_MIN
    h = h.drop(columns=["ex", "sl", "zone"]).reindex(range(int(h.index.max()) + 1))
    h["rest_hr"] = h.rest_hr.ffill().bfill()
    h[["motion", "env_load", "temp_c", "rh_pct"]] = h[["motion", "env_load", "temp_c", "rh_pct"]].ffill().bfill()
    h[["exercise_min", "sleep_min", "hr_zone_min"]] = h[["exercise_min", "sleep_min", "hr_zone_min"]].fillna(0)
    h["hod"] = h.index % 24
    return h

def analyse(h, baseline_hours=48, hr_scale=3.0):
    h = h.copy()
    b = h.iloc[:baseline_hours]
    # 1) personal expectation per hour-of-day (resting HR and motion)
    exp_hr = b.groupby("hod").rest_hr.median().reindex(range(24)).fillna(b.rest_hr.median())
    exp_mo = b.groupby("hod").motion.median().reindex(range(24)).fillna(b.motion.median())
    h["hr_resid"] = h.rest_hr - h.hod.map(exp_hr)
    h["motion_resid"] = h.motion - h.hod.map(exp_mo)
    br = h.hr_resid.iloc[:baseline_hours]
    scale = max(1.4826 * (br - br.median()).abs().median(), hr_scale)          # floor ~ normal day-to-day spread
    h["hr_z"] = h.hr_resid / scale
    h["hr_z_ewma"] = h.hr_z.ewm(span=3, adjust=False).mean()                   # persistent, not one-off, deviation
    # 2) multivariate outliers (needs >= 2 days of baseline to be meaningful)
    h["iso_flag"] = False
    if baseline_hours >= 48:
        from sklearn.ensemble import IsolationForest
        feats = ["hr_resid", "motion_resid", "env_load"]
        iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(h[feats].iloc[:baseline_hours])
        h["iso_flag"] = iso.predict(h[feats]) == -1
    # 3) readiness (trailing 24 h, 0-100)
    r = lambda c, f: h[c].rolling(24, min_periods=24).agg(f)
    h["sleep_score"] = (100 * (r("sleep_min", "sum") / 450).clip(0, 1))
    h["hr_score"] = 100 * (1 - (h.hr_z_ewma.clip(lower=0) / 4).clip(0, 1))
    ex24 = r("exercise_min", "sum")
    h["load_score"] = np.where(ex24 < 60, 100 * ex24 / 60, 100 * (1 - ((ex24 - 150) / 150).clip(0, 1)))
    h.loc[ex24.isna(), "load_score"] = np.nan
    h["env_score"] = 100 * (1 - (r("env_load", "mean") / 1.5).clip(0, 1))
    h["readiness"] = (W_SLEEP*h.sleep_score + W_HR*h.hr_score + W_LOAD*h.load_score + W_ENV*h.env_score).round(1)
    # 4) alerts with a plain-language reason
    z = h.hr_z_ewma
    h["alert"] = np.select([z >= 4, (z >= 2.5) | (h.iso_flag & (z >= 1.5))], ["RED", "AMBER"], "GREEN")
    comp = h[["sleep_score", "hr_score", "load_score", "env_score"]]
    weakest = comp.idxmin(axis=1).where(comp.notna().all(axis=1))
    reason = []
    for i in h.index:
        s = ""
        if h.alert[i] != "GREEN":
            s = f"resting HR {h.hr_resid[i]:+.0f} bpm vs personal baseline (persistent z={z[i]:.1f})"
            if h.iso_flag[i]: s += "; multivariate outlier"
        elif pd.notna(h.readiness[i]) and h.readiness[i] < 60:
            s = f"low readiness; weakest: {weakest[i].replace('_score','')}"
        reason.append(s)
    h["reason"] = reason
    return h

def evaluate(h, injected):
    flagged = set(h.index[h.alert != "GREEN"])
    tp = len(flagged & injected)
    prec = tp / len(flagged) if flagged else float("nan")
    rec = tp / len(injected) if injected else float("nan")
    first = [x for x in sorted(flagged) if x >= min(injected)]
    delay = (first[0] - min(injected)) if first else None
    return prec, rec, delay

def plot(h, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    col = {"GREEN": "tab:green", "AMBER": "orange", "RED": "red"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(h.index, h.rest_hr, color="grey", lw=1)
    ax[0].scatter(h.index, h.rest_hr, c=h.alert.map(col), s=18)
    ax[0].set_ylabel("resting-state HR (bpm)"); ax[0].set_title("AstroVital health timeline (green/amber/red = alert level)")
    ax[1].plot(h.index, h.readiness, color="tab:blue"); ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("readiness (0-100)"); ax[1].set_xlabel("mission hour")
    fig.tight_layout(); fig.savefig(out, dpi=150)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--baseline-hours", type=int, default=48)
    ap.add_argument("--inject", action="store_true", help="inject a synthetic HR anomaly to test detection")
    ap.add_argument("--inject-start", type=int, default=72)
    ap.add_argument("--inject-hours", type=int, default=6)
    ap.add_argument("--inject-delta", type=float, default=12.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = pd.read_csv(a.features)
    injected = set()
    if a.inject:
        w, injected = inject_hr_anomaly(w, a.inject_start, a.inject_hours, a.inject_delta)
    h = analyse(hourly_features(w), a.baseline_hours)
    h.to_csv(f"{a.out}/health_timeline.csv")
    plot(h, f"{a.out}/health_timeline.png")
    print(h.alert.value_counts().to_string())
    print(h.loc[h.alert != "GREEN", ["rest_hr", "hr_resid", "readiness", "alert", "reason"]].round(1).to_string())
    if injected:
        p, r, d = evaluate(h, injected)
        print(f"\nInjected anomaly hours {min(injected)}-{max(injected)}: precision={p:.2f} recall={r:.2f} detection delay={d} h")
    print(f"\nSaved to {a.out}/")

if __name__ == "__main__":
    main()
