"""Loading and validating the mission dataset."""
import pandas as pd
import config as C


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df, need_targets=True)
    return df.sort_values([C.GROUP_COL, C.DAY_COL]).reset_index(drop=True)


def validate(df: pd.DataFrame, need_targets: bool = False) -> None:
    need = list(C.REQUIRED_COLUMNS)
    if need_targets:
        need += [C.RISK_COL, C.READINESS_COL, C.ANOMALY_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[need].isna().any().any():
        bad = df[need].columns[df[need].isna().any()].tolist()
        raise ValueError(f"NaN values found in: {bad}")
    if df.duplicated([C.GROUP_COL, C.DAY_COL]).any():
        raise ValueError("Duplicate (crew_id, mission_day) rows found")
    if need_targets:
        unknown = set(df[C.RISK_COL]) - set(C.RISK_ORDER)
        if unknown:
            raise ValueError(f"Unknown risk labels: {unknown}")
