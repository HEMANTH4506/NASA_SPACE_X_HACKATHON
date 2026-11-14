"""Inference. Works on a DataFrame/CSV with the same columns as the training data.
Rows must include the astronaut's earlier days, because trend features look backwards.

CLI:  python predict.py data/nasa_astronaut_mission_full_dataset.csv --out reports/predictions.csv
      python predict.py <csv> --crew ASTRONAUT_02 --day 67
"""
import argparse

import joblib
import pandas as pd

import config as C
from data import validate
from features import add_features


def load_models():
    need = ["risk_today", "risk_next_day", "readiness"]
    missing = [n for n in need if not (C.MODEL_DIR / f"{n}.joblib").exists()]
    if missing:
        raise FileNotFoundError(f"Models not found {missing}. Run `python train.py` first.")
    return {n: joblib.load(C.MODEL_DIR / f"{n}.joblib") for n in need}


def predict_dataframe(raw: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    models = models or load_models()
    validate(raw)
    df = add_features(raw)

    b = models["risk_today"]
    proba = b["model"].predict_proba(df[b["num"] + b["cat"]])
    labels = b["labels"]
    df["pred_risk"] = [labels[i] for i in proba.argmax(1)]
    for i, lab in enumerate(labels):
        df[f"p_{lab.lower()}"] = proba[:, i].round(4)

    r = models["readiness"]
    df["pred_readiness_pct"] = r["model"].predict(df[r["num"] + r["cat"]]).clip(0, 100).round(1)

    n = models["risk_next_day"]
    if all(c in df.columns for c in n["num"] + n["cat"]):
        p = n["model"].predict_proba(df[n["num"] + n["cat"]])[:, 1]
        df["p_atrisk_tomorrow"] = p.round(4)
        df["early_warning"] = (p >= n["threshold"]).astype(int)
    return df.sort_index()


OUT_COLS = ["crew_id", "mission_day", "mission_phase", "pred_risk", "p_nominal", "p_warning", "p_critical",
            "pred_readiness_pct", "p_atrisk_tomorrow", "early_warning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out")
    ap.add_argument("--crew")
    ap.add_argument("--day", type=int)
    a = ap.parse_args()
    out = predict_dataframe(pd.read_csv(a.csv))
    cols = [c for c in OUT_COLS if c in out.columns]
    show = out
    if a.crew:
        show = show[show.crew_id == a.crew]
    if a.day is not None:
        show = show[show.mission_day == a.day]
    print(show[cols].head(30).to_string(index=False))
    if a.out:
        out[cols].to_csv(a.out, index=False)
        print(f"\nsaved {len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
