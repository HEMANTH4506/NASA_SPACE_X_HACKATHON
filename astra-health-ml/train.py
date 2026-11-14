"""Train and evaluate the ASTRA-HEALTH models.

Targets
  1. risk_today   : NOMINAL / WARNING / CRITICAL for the same day (no derived score columns used)
  2. risk_next    : will tomorrow be WARNING or CRITICAL?  (early-warning model)
  3. readiness    : hrp_readiness_index_pct regression (today, no leaky columns)

Validation = leave-one-astronaut-out: each fold trains on 3 crew members and tests on the
4th, so the score reflects performance on a person the model has never seen.

Run:  python train.py
"""
import json
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, mean_absolute_error,
                             r2_score, recall_score, roc_auc_score, average_precision_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from data import load_raw
from features import add_features, add_targets, feature_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
C.MODEL_DIR.mkdir(exist_ok=True)
C.REPORT_DIR.mkdir(exist_ok=True)


def preprocessor(num, cat, scale):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def classifiers(num, cat):
    rs = C.RANDOM_STATE
    return {
        "logreg": Pipeline([("p", preprocessor(num, cat, True)),
                            ("m", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))]),
        "random_forest": Pipeline([("p", preprocessor(num, cat, False)),
                                   ("m", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                                class_weight="balanced_subsample",
                                                                n_jobs=-1, random_state=rs))]),
        "hist_gb": Pipeline([("p", preprocessor(num, cat, False)),
                             ("m", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                                  max_iter=250, class_weight="balanced",
                                                                  random_state=rs))]),
    }


def regressors(num, cat):
    rs = C.RANDOM_STATE
    return {
        "ridge": Pipeline([("p", preprocessor(num, cat, True)), ("m", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([("p", preprocessor(num, cat, False)),
                                   ("m", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                               n_jobs=-1, random_state=rs))]),
        "hist_gb": Pipeline([("p", preprocessor(num, cat, False)),
                             ("m", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                                 max_iter=250, random_state=rs))]),
    }


def oof_predict(model, X, y, groups, proba=False):
    """Out-of-fold predictions with leave-one-astronaut-out."""
    out = None
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = __import__("sklearn.base", fromlist=["clone"]).clone(model).fit(X.iloc[tr], y.iloc[tr])
        if proba:
            p = m.predict_proba(X.iloc[te])
            if out is None:
                out = np.zeros((len(X), p.shape[1]))
            out[te] = p
        else:
            if out is None:
                out = np.zeros(len(X))
            out[te] = m.predict(X.iloc[te])
    return out


def plot_cm(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_importance(model, X, y, groups, scoring, n=12):
    """Permutation importance measured on held-out astronauts only."""
    imps = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = __import__("sklearn.base", fromlist=["clone"]).clone(model).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(m, X.iloc[te], y.iloc[te], scoring=scoring, n_repeats=5,
                                   random_state=C.RANDOM_STATE)
        imps.append(r.importances_mean)
    s = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
    return s.head(n)


def main():
    df = add_targets(add_features(load_raw()))
    groups_all = df[C.GROUP_COL]
    metrics = {"dataset": {"rows": len(df), "crew": int(groups_all.nunique()),
                           "class_counts": df[C.RISK_COL].value_counts().to_dict()}}

    # ------------------------------------------------------------------ 1. risk today
    num, cat = feature_columns(df, for_today=True)
    X, y, g = df[num + cat], df["risk_today"].astype(int), groups_all
    print(f"\n=== 1. risk_today  ({len(num)} numeric + {len(cat)} categorical features) ===")
    results, oofs = {}, {}
    for name, mdl in classifiers(num, cat).items():
        P = oof_predict(mdl, X, y, g, proba=True)
        pred = P.argmax(1)
        oofs[name] = (P, pred)
        results[name] = {"macro_f1": f1_score(y, pred, average="macro"),
                         "recall_warning_or_critical": recall_score(y >= 1, pred >= 1),
                         "recall_critical": recall_score(y == 2, pred == 2)}
        print(f"  {name:14s} macroF1={results[name]['macro_f1']:.3f}  "
              f"at-risk recall={results[name]['recall_warning_or_critical']:.3f}  "
              f"critical recall={results[name]['recall_critical']:.3f}")
    best = max(results, key=lambda k: results[k]["macro_f1"])
    P, pred = oofs[best]
    print(f"  -> best: {best}")
    print(classification_report(y, pred, target_names=C.RISK_ORDER, digits=3))
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    plot_cm(cm, C.RISK_ORDER, f"Risk today ({best}, leave-one-crew-out)", C.REPORT_DIR / "cm_risk_today.png")
    imp = top_importance(classifiers(num, cat)[best], X, y, g, "f1_macro")
    print("  top features:\n" + imp.round(4).to_string())
    metrics["risk_today"] = {
        "best_model": best, "all_models": results,
        "confusion_matrix(rows=actual,cols=pred)": cm.tolist(),
        "report": classification_report(y, pred, target_names=C.RISK_ORDER, output_dict=True),
        "top_features": imp.round(4).to_dict()}
    final = classifiers(num, cat)[best].fit(X, y)
    joblib.dump({"model": final, "num": num, "cat": cat, "labels": C.RISK_ORDER, "name": best},
                C.MODEL_DIR / "risk_today.joblib")

    # ------------------------------------------------------------------ 2. risk next day
    num2, cat2 = feature_columns(df, for_today=False)
    d2 = df[df["atrisk_next"].notna()]
    X2, y2, g2 = d2[num2 + cat2], d2["atrisk_next"].astype(int), d2[C.GROUP_COL]
    print(f"\n=== 2. risk_next (WARNING/CRITICAL tomorrow?)  rows={len(d2)}, positives={int(y2.sum())} ===")
    results2, oofs2 = {}, {}
    for name, mdl in classifiers(num2, cat2).items():
        P2 = oof_predict(mdl, X2, y2, g2, proba=True)[:, 1]
        oofs2[name] = P2
        results2[name] = {"roc_auc": roc_auc_score(y2, P2), "avg_precision": average_precision_score(y2, P2),
                          "f1@0.5": f1_score(y2, P2 >= 0.5)}
        print(f"  {name:14s} AUC={results2[name]['roc_auc']:.3f}  AP={results2[name]['avg_precision']:.3f}  "
              f"F1@0.5={results2[name]['f1@0.5']:.3f}")
    best2 = max(results2, key=lambda k: results2[k]["avg_precision"])
    P2 = oofs2[best2]
    # pick threshold that keeps >=80% recall (early-warning: missing an event is worse than a false alarm)
    ths = np.linspace(0.05, 0.95, 91)
    ok = [t for t in ths if recall_score(y2, P2 >= t) >= 0.80]
    thr = float(max(ok)) if ok else 0.5
    p2 = (P2 >= thr).astype(int)
    print(f"  -> best: {best2}; alert threshold={thr:.2f} (recall >= 80%)")
    print(classification_report(y2, p2, target_names=["ok_tomorrow", "at_risk_tomorrow"], digits=3))
    cm2 = confusion_matrix(y2, p2)
    plot_cm(cm2, ["ok", "at risk"], f"Risk tomorrow ({best2})", C.REPORT_DIR / "cm_risk_next.png")
    metrics["risk_next"] = {"best_model": best2, "all_models": results2, "alert_threshold": thr,
                            "confusion_matrix": cm2.tolist(),
                            "report": classification_report(y2, p2, output_dict=True)}
    final2 = classifiers(num2, cat2)[best2].fit(X2, y2)
    joblib.dump({"model": final2, "num": num2, "cat": cat2, "threshold": thr, "name": best2},
                C.MODEL_DIR / "risk_next_day.joblib")

    # ------------------------------------------------------------------ 3. readiness index
    yr = df[C.READINESS_COL]
    print("\n=== 3. readiness index regression ===")
    res3 = {}
    for name, mdl in regressors(num, cat).items():
        pr = oof_predict(mdl, X, yr, g)
        res3[name] = {"mae": mean_absolute_error(yr, pr), "r2": r2_score(yr, pr)}
        print(f"  {name:14s} MAE={res3[name]['mae']:.2f} pts   R2={res3[name]['r2']:.3f}")
    best3 = min(res3, key=lambda k: res3[k]["mae"])
    print(f"  -> best: {best3}")
    metrics["readiness"] = {"best_model": best3, "all_models": res3}
    final3 = regressors(num, cat)[best3].fit(X, yr)
    joblib.dump({"model": final3, "num": num, "cat": cat, "name": best3}, C.MODEL_DIR / "readiness.joblib")

    with open(C.REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nSaved models -> {C.MODEL_DIR}\nSaved reports -> {C.REPORT_DIR}")


if __name__ == "__main__":
    main()
