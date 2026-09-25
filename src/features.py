"""Feature engineering: aggregate per-run CI history into per-test features.

Each feature is chosen to separate *flaky* tests (nondeterministic) from
*stable* tests (fail only on real regressions):

- fail_rate            fraction of runs that failed
- flip_rate            fraction of consecutive runs with a changed outcome
                       (flaky tests oscillate pass/fail/pass)
- duration_mean/std/cv run-time statistics; flaky tests have unstable timing
- churn_fail_ratio     avg churn on failed runs / avg churn on passed runs.
                       Stable tests fail on high churn (ratio >> 1); flaky
                       tests fail regardless of churn (ratio ~= 1).
- fail_churn_corr      correlation between failure and churn; near 0 for flaky
- num_assertions, age_days  test metadata
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

FEATURE_COLS = [
    "fail_rate", "flip_rate", "duration_mean", "duration_std", "duration_cv",
    "churn_fail_ratio", "fail_churn_corr", "num_assertions", "age_days",
]


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for test_id, grp in history.groupby("test_id"):
        grp = grp.sort_values("run_id")
        passed = grp["passed"].to_numpy()
        durations = grp["duration_s"].to_numpy()
        churn = grp["churn_lines"].to_numpy()

        fail_rate = 1.0 - passed.mean()
        flip_rate = float(np.mean(passed[1:] != passed[:-1])) if len(passed) > 1 else 0.0
        duration_mean = float(durations.mean())
        duration_std = float(durations.std())
        duration_cv = duration_std / duration_mean if duration_mean > 0 else 0.0

        failed_mask = passed == 0
        churn_on_fail = churn[failed_mask].mean() if failed_mask.any() else 0.0
        churn_on_pass = churn[~failed_mask].mean() if (~failed_mask).any() else 0.0
        churn_fail_ratio = churn_on_fail / (churn_on_pass + 1e-6)
        # point-biserial style correlation between failure (0/1) and churn
        failed01 = 1 - passed
        if failed01.std() > 0 and churn.std() > 0:
            fail_churn_corr = float(np.corrcoef(failed01, churn)[0, 1])
        else:
            fail_churn_corr = 0.0

        feats.append({
            "test_id": test_id,
            "fail_rate": fail_rate,
            "flip_rate": flip_rate,
            "duration_mean": duration_mean,
            "duration_std": duration_std,
            "duration_cv": duration_cv,
            "churn_fail_ratio": churn_fail_ratio,
            "fail_churn_corr": fail_churn_corr,
            "num_assertions": int(grp["num_assertions"].iloc[0]),
            "age_days": int(grp["age_days"].iloc[0]),
        })
    return pd.DataFrame(feats)


def load_labeled() -> pd.DataFrame:
    history = pd.read_csv(os.path.join(DATA_DIR, "test_history.csv"))
    labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
    feats = build_features(history)
    return feats.merge(labels, on="test_id")


if __name__ == "__main__":
    df = load_labeled()
    print(df.head().to_string())
    print(f"\n{len(df)} tests x {len(FEATURE_COLS)} features")
