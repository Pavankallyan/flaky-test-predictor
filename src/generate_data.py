"""Generate synthetic CI test-history data with a known set of flaky tests.

Simulates N_TESTS automated tests, each executed across N_RUNS CI runs.
A fraction of tests are *flaky*: they fail intermittently regardless of code
changes and show high run-duration variance. Stable tests only fail when the
code churn they cover is high (real regressions) and have steady durations.

Outputs:
    data/test_history.csv  - one row per (test, run): durations, pass/fail, churn
    data/labels.csv        - ground-truth flaky flag per test (from triage)
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_TESTS = 300
N_RUNS = 60
FLAKY_FRACTION = 0.15

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


def generate():
    history_rows = []
    label_rows = []
    for test_id in range(N_TESTS):
        is_flaky = RNG.random() < FLAKY_FRACTION
        base_duration = float(RNG.lognormal(mean=1.5, sigma=0.8))  # seconds
        num_assertions = int(RNG.integers(3, 40))
        age_days = int(RNG.integers(30, 900))
        suite = RNG.choice(["unit", "api", "ui", "integration"])
        if is_flaky:
            # flaky tests span a spectrum: some barely flaky, some wild.
            # Deliberate overlap with stable tests on both axes.
            fail_p = float(RNG.uniform(0.03, 0.20))
            dur_sigma = float(RNG.uniform(0.25, 1.00))
        else:
            fail_p, dur_sigma = 0.0, 0.15
            if RNG.random() < 0.15:
                # borderline-stable: mildly intermittent with wobbly timing,
                # genuinely hard to tell apart from mildly flaky tests
                fail_p, dur_sigma = 0.03, float(RNG.uniform(0.30, 0.60))

        for run_id in range(N_RUNS):
            # code churn (lines changed in files this test covers); usually 0
            churn = float(RNG.exponential(scale=15.0)) if RNG.random() < 0.3 else 0.0
            if is_flaky or fail_p > 0:
                # intermittent failures uncorrelated with churn + unstable timing
                failed = RNG.random() < fail_p
                duration = base_duration * float(RNG.lognormal(0.0, dur_sigma))
            else:
                # stable tests fail only on real regressions (high churn)
                failed = RNG.random() < min(0.02 + churn / 400.0, 0.5)
                duration = base_duration * float(RNG.lognormal(0.0, dur_sigma))
            history_rows.append({
                "test_id": f"test_{test_id:03d}",
                "run_id": run_id,
                "suite": suite,
                "duration_s": round(duration, 3),
                "passed": int(not failed),
                "churn_lines": round(churn, 1),
                "num_assertions": num_assertions,
                "age_days": age_days,
            })
        label_rows.append({"test_id": f"test_{test_id:03d}", "is_flaky": int(is_flaky)})

    os.makedirs(DATA_DIR, exist_ok=True)
    history = pd.DataFrame(history_rows)
    labels = pd.DataFrame(label_rows)
    history.to_csv(os.path.join(DATA_DIR, "test_history.csv"), index=False)
    labels.to_csv(os.path.join(DATA_DIR, "labels.csv"), index=False)
    print(f"wrote {len(history)} run records for {N_TESTS} tests "
          f"({labels['is_flaky'].sum()} flaky) -> {DATA_DIR}")


if __name__ == "__main__":
    generate()
