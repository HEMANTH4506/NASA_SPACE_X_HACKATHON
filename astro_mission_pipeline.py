"""
Synthetic ISS-style mission timeline + activity recognition + environmental-context analytics.

Pipeline:
 1. simulate multi-day crew timeline (sleep/work/meal/exercise) with IMU, HR, temp, humidity, light
 2. window the signals (30 s) and extract features
 3. activity recognition + intensity estimation (group-by-day cross-validation)
 4. align activity / environment / HR and build mission-context timeline features

NOTE: the data is SIMULATED. Replace simulate_mission() with a loader for PAMAP2 / PPG-DaLiA /
LSDA / analog-mission data for real validation.
Usage: python astro_mission_pipeline.py --days 3 --gravity micro --out out/
"""
import argparse, os
import numpy as np, pandas as pd

FS, WIN = 10, 30                      # Hz, window seconds
HR_REST, HR_MAX = 60, 190
ORBIT_S = 92 * 60                     # ISS orbit ~92 min

# label: (acc amp g, freq Hz, gyro amp deg/s, target HR bpm)
ACT = {
    "sleep":     (0.01, 0.2, 2,   54),
    "rest":      (0.04, 0.3, 5,   66),
    "work":      (0.08, 0.5, 10,  72),
    "meal":      (0.10, 0.8, 12,  74),
    "treadmill": (1.20, 2.6, 120, 152),
    "cycling":   (0.40, 1.4, 60,  132),
    "resistive": (0.80, 0.4, 70,  122),
}
TEMPLATE = [(0,390,"sleep"),(390,420,"meal"),(420,540,"work"),(540,600,"treadmill"),
            (600,630,"rest"),(630,720,"work"),(720,750,"meal"),(750,870,"work"),
            (870,930,"resistive"),(930,960,"rest"),(960,1080,"work"),(1080,1125,"cycling"),
            (1125,1170,"rest"),(1170,1200,"meal"),(1200,1350,"work"),(1350,1440,"sleep")]

def smooth(x, sec):
    return pd.Series(x).ewm(span=sec * FS, adjust=False).mean().to_numpy()

def simulate_mission(days=3, gravity="micro", seed=0):
    rng = np.random.default_rng(seed)
    g0 = 0.0 if gravity == "micro" else 1.0
    seg_lab, ax, ay, az, gx, gy, gz, hr_t = [], [], [], [], [], [], [], []
    for d in range(days):
        for s, e, lab in TEMPLATE:
            if lab == "resistive" and d % 2 == 1:
                lab = "rest"                                  # resistive exercise on alternate days
            n = (e - s) * 60 * FS
            amp, f, gamp, hr = ACT[lab]
            t = np.arange(n) / FS
            ph = rng.uniform(0, 2*np.pi, 3)
            noise = 0.02 + 0.1 * amp
            ax.append(amp*np.sin(2*np.pi*f*t+ph[0]) + rng.normal(0, noise, n))
            ay.append(0.6*amp*np.sin(2*np.pi*f*t+ph[1]) + rng.normal(0, noise, n))
            az.append(g0 + 0.8*amp*np.sin(2*np.pi*2*f*t+ph[2]) + rng.normal(0, noise, n))
            gx.append(gamp*np.sin(2*np.pi*f*t+ph[0]) + rng.normal(0, 2, n))
            gy.append(0.7*gamp*np.sin(2*np.pi*f*t+ph[1]) + rng.normal(0, 2, n))
            gz.append(0.5*gamp*np.sin(2*np.pi*f*t+ph[2]) + rng.normal(0, 2, n))
            hr_t.append(np.full(n, hr + rng.normal(0, 3)))
            seg_lab.append(np.full(n, lab, dtype=object))
    cat = lambda a: np.concatenate(a)
    label, N = cat(seg_lab), sum(len(a) for a in ax)
    t_s = np.arange(N) / FS
    hr = smooth(cat(hr_t), 45) + rng.normal(0, 1.2, N)
    ex = np.isin(label, ["treadmill", "cycling", "resistive"]).astype(float)
    hour = (t_s / 3600) % 24
    temp = 22.5 + 1.0*np.sin(2*np.pi*(hour-14)/24) + smooth(rng.normal(0, 0.4, N), 600) + 1.5*smooth(ex, 300)
    hum = 45 + 5*np.sin(2*np.pi*(hour-6)/24) + smooth(rng.normal(0, 1.0, N), 600) + 8*smooth(ex, 240)
    awake = (label != "sleep").astype(float)
    phase = (t_s % ORBIT_S) / ORBIT_S
    sunlit = (phase < 0.6).astype(float)
    lux = (2 + 400*awake) * np.exp(rng.normal(0, 0.15, N)) + sunlit * awake * rng.uniform(50, 250)
    return pd.DataFrame({
        "t_s": t_s, "label": label,
        "acc_x": cat(ax), "acc_y": cat(ay), "acc_z": cat(az),
        "gyr_x": cat(gx), "gyr_y": cat(gy), "gyr_z": cat(gz),
        "hr": hr, "temp_c": temp, "rh_pct": np.clip(hum, 10, 90), "lux": lux,
        "orbit_phase": phase, "sunlit": sunlit,
    })

def window_features(df):
    spw = WIN * FS
    n = len(df) // spw
    d = df.iloc[: n * spw]
    R = lambda c: d[c].to_numpy().reshape(n, spw)
    mag = np.sqrt(R("acc_x")**2 + R("acc_y")**2 + R("acc_z")**2)
    gmag = np.sqrt(R("gyr_x")**2 + R("gyr_y")**2 + R("gyr_z")**2)
    spec = np.abs(np.fft.rfft(mag - mag.mean(1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(spw, 1/FS)
    hr = R("hr")
    t0 = d["t_s"].to_numpy().reshape(n, spw)[:, 0]
    w = pd.DataFrame({
        "t_s": t0, "day": (t0 // 86400).astype(int),
        "label": d["label"].to_numpy().reshape(n, spw)[:, 0],
        "acc_mean": mag.mean(1), "acc_std": mag.std(1),
        "acc_mad": np.abs(mag - mag.mean(1, keepdims=True)).mean(1),
        "acc_x_std": R("acc_x").std(1), "acc_y_std": R("acc_y").std(1), "acc_z_std": R("acc_z").std(1),
        "gyr_rms": np.sqrt((gmag**2).mean(1)),
        "dom_freq": freqs[spec[:, 1:].argmax(1) + 1], "spec_energy": (spec**2).sum(1),
        "hr_mean": hr.mean(1), "hr_max": hr.max(1), "hr_slope": hr[:, -1] - hr[:, 0],
        "temp_c": R("temp_c").mean(1), "rh_pct": R("rh_pct").mean(1),
        "lux_log": np.log10(R("lux").mean(1) + 1), "sunlit_frac": R("sunlit").mean(1),
        "orbit_phase": R("orbit_phase")[:, 0],
    })
    w["orbit_sin"] = np.sin(2*np.pi*w.orbit_phase); w["orbit_cos"] = np.cos(2*np.pi*w.orbit_phase)
    hrr = ((w.hr_mean - HR_REST) / (HR_MAX - HR_REST)).clip(0, 1)
    w["hrr"] = hrr
    w["intensity"] = pd.cut(hrr, [-1, .2, .4, .6, 2], labels=["sedentary", "light", "moderate", "vigorous"]).astype(str)
    return w

MOTION_HR = ["acc_mean","acc_std","acc_mad","acc_x_std","acc_y_std","acc_z_std","gyr_rms",
             "dom_freq","spec_energy","hr_mean","hr_max","hr_slope"]
MOTION_ONLY = [c for c in MOTION_HR if not c.startswith("hr_")]

def cv_predict(w, cols, target):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    k = min(5, w.day.nunique())
    clf = RandomForestClassifier(200, random_state=0, n_jobs=-1)
    return cross_val_predict(clf, w[cols], w[target], groups=w.day, cv=GroupKFold(k))

def context_features(w, act_col="act_pred"):
    """Mission-context timeline built from PREDICTED activity (no label leakage)."""
    w = w.copy()
    exercise = w[act_col].isin(["treadmill", "cycling", "resistive"])
    sleep = w[act_col] == "sleep"
    w["sleep_cycle"] = ((~sleep) & sleep.shift(1, fill_value=True)).cumsum()      # increments at each wake-up
    w["exercise_min_cycle"] = exercise.groupby(w.sleep_cycle).cumsum() * WIN / 60
    zone = (w.hrr >= 0.4) & exercise
    w["hr_zone_min_cycle"] = zone.groupby(w.sleep_cycle).cumsum() * WIN / 60
    last = w.t_s.where(exercise).ffill()
    w["min_since_exercise"] = ((w.t_s - last) / 60).fillna(np.nan)
    w["hours_awake"] = (~sleep).groupby(w.sleep_cycle).cumsum() * WIN / 3600
    r = int(30 * 60 / WIN)                                                         # 30-min rolling exposure
    for c in ["temp_c", "rh_pct", "lux_log"]:
        w[f"{c}_roll30"] = w[c].rolling(r, min_periods=1).mean()
    w["env_load_index"] = ((w.temp_c - 22.5).abs()/3 + (w.rh_pct - 45).abs()/20).rolling(r, min_periods=1).mean()
    return w

def aligned_events(w):
    """One row per contiguous activity bout with its environment + HR summary."""
    b = (w.act_pred != w.act_pred.shift()).cumsum()
    return w.groupby(b).agg(start_s=("t_s","first"), end_s=("t_s","last"), activity=("act_pred","first"),
        minutes=("t_s", lambda s: len(s)*WIN/60), hr_mean=("hr_mean","mean"), hr_peak=("hr_max","max"),
        hrr=("hrr","mean"), temp_c=("temp_c","mean"), rh_pct=("rh_pct","mean"),
        lux_log=("lux_log","mean")).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--gravity", choices=["micro", "earth"], default="micro")
    ap.add_argument("--out", default="out")
    ap.add_argument("--save-raw", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = simulate_mission(a.days, a.gravity)
    if a.save_raw: df.to_csv(f"{a.out}/raw_timeline.csv.gz", index=False)
    w = window_features(df)

    from sklearn.metrics import classification_report
    w["act_pred"] = cv_predict(w, MOTION_HR, "label")
    w["int_pred"] = cv_predict(w, MOTION_ONLY, "intensity")        # intensity without HR (HR may be missing)
    print("== Activity recognition (leave-day-out) ==")
    print(classification_report(w.label, w.act_pred, digits=3))
    print("== Intensity from motion only ==")
    print(classification_report(w.intensity, w.int_pred, digits=3))

    w = context_features(w)
    ev = aligned_events(w)
    w.to_csv(f"{a.out}/window_features_context.csv", index=False)
    ev.to_csv(f"{a.out}/aligned_events.csv", index=False)
    per_cycle = w.groupby("sleep_cycle").agg(exercise_min=("exercise_min_cycle","max"),
        hr_zone_min=("hr_zone_min_cycle","max"), env_load=("env_load_index","mean"),
        hours_awake=("hours_awake","max"))
    per_cycle.to_csv(f"{a.out}/per_sleep_cycle_summary.csv")
    print(per_cycle.round(2))
    print(f"\nSaved to {a.out}/")

if __name__ == "__main__":
    main()
