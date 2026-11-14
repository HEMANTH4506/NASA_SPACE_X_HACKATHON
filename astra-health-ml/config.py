"""Central settings: paths, column groups, and which columns are allowed as model inputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "nasa_astronaut_mission_full_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42

# ---- identifiers / targets ----
GROUP_COL = "crew_id"          # used to split by astronaut, never as a feature
DAY_COL = "mission_day"
RISK_COL = "operational_risk_status"
READINESS_COL = "hrp_readiness_index_pct"
ANOMALY_COL = "persistent_fatigue_anomaly"
RISK_ORDER = ["NOMINAL", "WARNING", "CRITICAL"]   # ordinal: 0 < 1 < 2

# ---- raw inputs ----
CONTEXT_NUM = ["rest_day_flag", "eva_scheduled", "radiation_alert", "circadian_shift_hrs",
               "exercise_duration_mins", "workload_score_1_10"]
PHYSIO_NUM = ["sleep_duration_hrs", "sleep_efficiency_pct", "cumulative_sleep_debt_hrs",
              "resting_hr_bpm", "hrv_rmssd_ms", "spo2_pct", "core_body_temp_c",
              "pvt_reaction_time_ms", "pvt_attention_lapses",
              "subjective_fatigue_1_10", "subjective_stress_1_10"]
BASELINE_NUM = ["baseline_sleep_14d_mean", "baseline_hr_14d_mean",
                "baseline_hrv_14d_mean", "baseline_pvt_14d_mean", "sleep_z_score"]
CATEGORICAL = ["mission_phase"]

# Columns that are DERIVED OUTPUTS of the same scoring logic as the risk label.
# Using them to predict today's risk would be label leakage, so the "today" models
# never see them. (The next-day model may use them: they are known by end of today.)
LEAKY_FOR_TODAY = [READINESS_COL, ANOMALY_COL]

REQUIRED_COLUMNS = (["crew_id", DAY_COL] + CONTEXT_NUM + PHYSIO_NUM + BASELINE_NUM + CATEGORICAL)
