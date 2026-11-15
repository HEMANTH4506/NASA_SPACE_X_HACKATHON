# AstroVital: Astronaut Health Monitoring Software

NASA Space Apps Challenge 2026 | Problem statement: *Create Health Monitoring Software for Astronauts on Space Missions*

AstroVital is an edge-first health monitoring system that fuses wearable physiology, activity and cabin-environment data into a personal readiness score, so crews and ground teams can spot health risks early, even when Earth is minutes away.

**Live demo:** [add your link, or "run locally with the commands below"]
**Demo video:** [add your video link]

## Why it matters

Long missions bring microgravity effects, disrupted sleep, heavy exercise demands, radiation and confined living. On deep-space missions, communication delays make real-time ground support impossible, so monitoring has to work on board with little bandwidth.

## What it does

1. **Activity recognition:** classifies sleep, work, rest, treadmill, cycling and resistive exercise from accelerometer, gyroscope and heart rate.
2. **Intensity and exercise load:** heart-rate-reserve zones, and minutes in zone per sleep cycle.
3. **Personal baseline and anomaly detection:** learns each crew member's normal resting heart rate by hour of day, then flags persistent deviations (EWMA z-score) and multivariate outliers (Isolation Forest).
4. **Readiness score (0-100):** sleep, heart-rate deviation, exercise load and cabin environment, with visible weights.
5. **Alerts:** GREEN / AMBER / RED, each with a plain-language reason.
6. **Dashboard (Streamlit):** crew view, ground-team view, mission replay and an anomaly simulator.

## Architecture

```
Sensing -> Edge processing -> Analytics engine -> Alerts and decision support -> Dashboard -> Ground sync
```

## Repository layout

```
astro_mission_pipeline.py   # simulated crew timeline, features, activity/intensity models, context features
astro_health_module.py      # personal baseline, anomaly detection, readiness score, alerts
nasa_env_integration.py     # replaces the simulated cabin environment with real ISS telemetry
nasa_osdr_fetch.py          # search and download NASA OSDR studies
nasa_iss_live_logger.py     # log live ISS cabin telemetry (NASA ISS Live feed)
astrovital_dashboard.py     # Streamlit dashboard
health_timeline.json        # hourly output (for web front ends)
data/nasa/eda_rr1.csv       # real ISS cabin temperature, humidity and CO2 (NASA OSDR EDA)
astra-health-ml/            # [describe: base project this repo was forked from]
```

## Quick start

```bash
pip install numpy pandas scikit-learn matplotlib streamlit altair requests

# 1. Simulated 5-day crew timeline
python astro_mission_pipeline.py --days 5 --out out/

# 2. Replace simulated cabin conditions with real NASA ISS telemetry
python nasa_env_integration.py --eda data/nasa/eda_rr1.csv --features out/window_features_context.csv --out out/

# 3. Baseline, anomaly detection, readiness and alerts (with a synthetic test anomaly)
python astro_health_module.py --features out/window_features_real_env.csv --inject --out out/

# 4. Dashboard
streamlit run astrovital_dashboard.py
```

## Data sources

| Source | Use | Status |
|---|---|---|
| NASA OSDR Environmental Data App, mission RR-1 (12,087 ISS cabin readings, 2014-09-20 to 2014-11-01) | Real cabin temperature (mean 22.8 °C), humidity (mean 44%) and CO2 (mean about 3,200 ppm; ground air is about 400-500 ppm) | **Real NASA data** |
| NASA ISS Live feed (cabin pressure, USLAB000058) | Exploratory live logging with `nasa_iss_live_logger.py` | Real, single logging session |
| Crew activity, heart rate and motion signals | Demo timeline | **Simulated** |
| PAMAP2 (Reiss & Stricker, DFKI; hosted by UCI) | Real-data validation of the activity model | [Used / Planned] |

**Citation:** A. Reiss and D. Stricker, "Introducing a New Benchmarked Dataset for Activity Monitoring," ISWC 2012.

## Results

| Test | Result |
|---|---|
| Anomaly detection on a simulated injected heart-rate anomaly (6 h, +12 bpm) | 5 of 6 hours flagged, no false alarms, first alert 1 h after onset |
| Activity recognition on real PAMAP2 data (leave-one-subject-out) | [fill in after running the PAMAP2 pipeline, or delete this row] |

The anomaly result comes from simulated data and is not real-world accuracy.

## Limitations

- Crew activity, heart rate and motion are simulated; the cabin environment is real NASA ISS data from 2014 (mission RR-1), looped over the simulated mission.
- Spaceflight validation would need NASA Life Sciences Data Archive or analog-mission data.
- Readiness weights and alert thresholds are starting values, not clinically validated.
- AstroVital is decision support, not a medical device, and it does not diagnose.

## Roadmap

- Validate on analog-mission data (for example HERA, SIRIUS, HI-SEAS)
- Use CO2 as a readiness component and alert
- Add sleep staging and radiation-exposure tracking
- Run on-board on edge hardware, with priority-based ground sync

## Team

[Names and roles]

## AI tools used

[Describe the AI tools used and for what, as the Space Apps rules require.]

## License

[Choose a license, for example MIT]
