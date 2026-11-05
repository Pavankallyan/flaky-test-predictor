"""Walkthrough: score a brand-new test suite with the trained model.

Builds a small 3-test suite by hand (one rock-stable, one borderline, one
wildly flaky), then runs src/score.py on it so you can see the ranking and
the verdicts without touching the real history in data/.

Usage (from the repo root):
    python examples/score_walkthrough.py

Trains the model first (python src/train.py) if models/flaky_model.pkl
does not exist yet.
"""
import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "models", "flaky_model.pkl")

COLS = ["test_id", "run_id", "suite", "duration_s", "passed",
        "churn_lines", "num_assertions", "age_days"]


def stable_rows(test_id, n=12, duration=8.0):
    """Always passes, steady timing, low churn: the suite's reliable workhorse."""
    return [(test_id, i, "unit", duration, 1, 5, 12, 200) for i in range(n)]


def borderline_rows(test_id, n=12):
    """Fails twice in 12 runs on quiet days, timing a bit wobbly: the test a
    QA lead would argue about in triage."""
    return [(test_id, i, "api", 10.0 + (i % 3), 0 if i in (4, 9) else 1, 5, 8, 90)
            for i in range(n)]


def flaky_rows(test_id, n=12):
    """Oscillates pass/fail and its duration swings 5x run to run, with no
    link to code churn: the classic flaky signature."""
    return [(test_id, i, "ui", 4.0 + 4.0 * (i % 5), 1 if i % 2 == 0 else 0,
             5, 20, 400) for i in range(n)]


def main():
    if not os.path.exists(MODEL_PATH):
        print("no trained model yet - training first (this takes a minute)...")
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "train.py")],
                       check=True, cwd=ROOT)
        print()

    rows = (stable_rows("test_checkout_total")
            + borderline_rows("test_search_filters")
            + flaky_rows("test_payment_retry"))
    history = pd.DataFrame(rows, columns=COLS)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "new_suite_history.csv")
        history.to_csv(csv_path, index=False)
        print(f"scoring a new 3-test suite from {csv_path}\n", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "score.py"),
                        csv_path], check=True, cwd=ROOT)

    print("\nWhat to do with the result:")
    print("- LIKELY FLAKY tests go to the top of the triage queue: quarantine")
    print("  them (or mark them non-blocking) until the root cause is found,")
    print("  so one flaky test stops failing the whole pipeline.")
    print("- Borderline tests are worth a look at their duration history:")
    print("  timing instability is the strongest flakiness signal here.")
    print("- Stable tests need nothing: they only fail on real regressions.")


if __name__ == "__main__":
    main()
