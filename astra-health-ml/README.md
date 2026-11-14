# ASTRA-HEALTH - ML backend (pure Python)

No FastAPI, database, or front end. CSV in -> trained models -> predictions.

## Run
    pip install -r requirements.txt
    python train.py                       # trains + evaluates, writes models/ and reports/
    python predict.py data/nasa_astronaut_mission_full_dataset.csv --out reports/predictions.csv
    python predict.py data/nasa_astronaut_mission_full_dataset.csv --crew ASTRONAUT_02 --day 67
    streamlit run app.py                  # optional quick UI

## What is predicted
| Model | Target | Notes |
|---|---|---|
| `risk_today` | NOMINAL / WARNING / CRITICAL | main requirement; readiness + anomaly columns excluded (they are derived from the same scoring, i.e. leakage) |
| `readiness` | hrp_readiness_index_pct | regression |
| `risk_next_day` | WARNING/CRITICAL tomorrow? | early-warning; threshold tuned for >=80% recall |

Validation is leave-one-astronaut-out, so scores are for crew the model never saw.
Metrics: `reports/metrics.json`, confusion matrices: `reports/*.png`.

## Files
config.py (columns/paths) · data.py (load+validate) · features.py (past-only rolling features) ·
train.py · predict.py · app.py

## Limits
Only 720 rows, 4 crew, 8 CRITICAL days -> CRITICAL numbers are very uncertain. The data looks
synthetic/rule-generated, so scores will not transfer to real astronauts. Not for medical use.
