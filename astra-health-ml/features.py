"""Feature engineering. Every rolling feature only looks at the PAST within one astronaut,
so a prediction for day t never uses information from day t+1 or later."""
import numpy as np
import pandas as pd
import config as C

ROLL_COLS = ["sleep_duration_hrs", "hrv_rmssd_ms", "resting_hr_bpm", "pvt_reaction_time_ms",
             "subjective_fatigue_1_10", "subjective_stress_1_10", "workload_score_1_10"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([C.GROUP_COL, C.DAY_COL]).copy()
    g = df.groupby(C.GROUP_COL, sort=False)

    # deviation from the astronaut's own 14-day baseline
    df["d_sleep_vs_base"] = df["sleep_duration_hrs"] - df["baseline_sleep_14d_mean"]
    df["d_hr_vs_base"] = df["resting_hr_bpm"] - df["baseline_hr_14d_mean"]
    df["d_hrv_vs_base"] = df["hrv_rmssd_ms"] - df["baseline_hrv_14d_mean"]
    df["d_pvt_vs_base"] = df["pvt_reaction_time_ms"] - df["baseline_pvt_14d_mean"]

    # short-term trends (past only)
    for col in ROLL_COLS:
        r3 = g[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        r7 = g[col].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{col}_r3"] = r3
        df[f"{col}_r7"] = r7
        df[f"{col}_chg1"] = g[col].diff().fillna(0.0)
    df["sleep_debt_chg1"] = g["cumulative_sleep_debt_hrs"].diff().fillna(0.0)

    # consecutive short-sleep nights
    short = (df["sleep_duration_hrs"] < 6.5).astype(int)
    run = short.groupby(df[C.GROUP_COL], sort=False).transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum())
    df["short_sleep_streak"] = run

    # combined-stressor flag
    df["stress_combo"] = df["eva_scheduled"] + df["radiation_alert"] + (df["circadian_shift_hrs"] > 1).astype(int)
    return df


def feature_columns(df: pd.DataFrame, for_today: bool) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for a model."""
    base = C.CONTEXT_NUM + C.PHYSIO_NUM + C.BASELINE_NUM
    if not for_today:                       # next-day model may also use today's derived scores
        base += C.LEAKY_FOR_TODAY
    engineered = [c for c in df.columns
                  if c.startswith("d_") or c.endswith(("_r3", "_r7", "_chg1"))
                  or c in ("sleep_debt_chg1", "short_sleep_streak", "stress_combo")]
    return base + engineered + [C.DAY_COL], list(C.CATEGORICAL)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_today"] = df[C.RISK_COL].map({k: i for i, k in enumerate(C.RISK_ORDER)})
    nxt = df.groupby(C.GROUP_COL, sort=False)["risk_today"].shift(-1)
    df["risk_next"] = nxt
    df["atrisk_next"] = (nxt >= 1).astype("float").where(nxt.notna())   # WARNING or CRITICAL tomorrow
    return df
