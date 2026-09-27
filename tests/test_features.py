"""Unit tests for src/features.py — feature engineering for the flaky-test predictor.

Tests use small synthetic CI histories with hand-computable expected values,
so each asserted number can be checked by reading the test.
"""
import os
import sys

import pandas as pd
import pytest

# repo root on sys.path so `from src.features import ...` resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURE_COLS, build_features


def make_history(rows):
    """rows: list of (test_id, run_id, passed, duration_s, churn_lines,
    num_assertions, age_days)"""
    cols = ["test_id", "run_id", "passed", "duration_s", "churn_lines",
            "num_assertions", "age_days"]
    return pd.DataFrame(rows, columns=cols)


def one_row(test_id, run_id, passed, duration=10.0, churn=5, assertions=4, age=30):
    return (test_id, run_id, passed, duration, churn, assertions, age)


def test_output_schema_matches_feature_cols():
    hist = make_history([one_row("t1", 1, 1), one_row("t1", 2, 1)])
    feats = build_features(hist)
    assert list(feats.columns) == ["test_id"] + FEATURE_COLS
    assert len(feats) == 1


def test_fail_rate_counts_failures():
    # 1 failure out of 4 runs -> 0.25
    hist = make_history([
        one_row("t1", 1, 1), one_row("t1", 2, 1),
        one_row("t1", 3, 0), one_row("t1", 4, 1),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "fail_rate"] == pytest.approx(0.25)


def test_flip_rate_detects_oscillation():
    # classic flaky signature: pass/fail/pass/fail -> every transition flips
    hist = make_history([
        one_row("t1", 1, 1), one_row("t1", 2, 0),
        one_row("t1", 3, 1), one_row("t1", 4, 0),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "flip_rate"] == pytest.approx(1.0)


def test_flip_rate_zero_for_stable_test():
    # stable test: no outcome changes (passes until the one real regression)
    hist = make_history([
        one_row("t1", 1, 1), one_row("t1", 2, 1),
        one_row("t1", 3, 1), one_row("t1", 4, 0),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "flip_rate"] == pytest.approx(1 / 3)


def test_single_run_has_no_flips_and_zero_spread():
    hist = make_history([one_row("t1", 1, 1, duration=12.0)])
    feats = build_features(hist)
    row = feats.loc[0]
    assert row["flip_rate"] == 0.0
    assert row["duration_std"] == pytest.approx(0.0)
    assert row["duration_cv"] == pytest.approx(0.0)
    assert row["duration_mean"] == pytest.approx(12.0)


def test_duration_stats():
    # durations 10, 20, 30 -> mean 20, std (population) = sqrt(200/3)
    hist = make_history([
        one_row("t1", 1, 1, duration=10.0),
        one_row("t1", 2, 1, duration=20.0),
        one_row("t1", 3, 1, duration=30.0),
    ])
    feats = build_features(hist)
    row = feats.loc[0]
    assert row["duration_mean"] == pytest.approx(20.0)
    assert row["duration_std"] == pytest.approx((200 / 3) ** 0.5)
    assert row["duration_cv"] == pytest.approx(((200 / 3) ** 0.5) / 20.0)


def test_churn_fail_ratio_high_for_stable_regression_failure():
    # stable test: fails only on the high-churn run -> ratio >> 1
    hist = make_history([
        one_row("t1", 1, 1, churn=5), one_row("t1", 2, 1, churn=5),
        one_row("t1", 3, 1, churn=5), one_row("t1", 4, 0, churn=50),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "churn_fail_ratio"] == pytest.approx(50 / 5)


def test_churn_fail_ratio_near_one_for_flaky_test():
    # flaky test: fails on churn identical to passing runs -> ratio ~= 1
    hist = make_history([
        one_row("t1", 1, 1, churn=10), one_row("t1", 2, 0, churn=10),
        one_row("t1", 3, 1, churn=10), one_row("t1", 4, 0, churn=10),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "churn_fail_ratio"] == pytest.approx(1.0)


def test_churn_fail_ratio_zero_when_no_failures():
    hist = make_history([
        one_row("t1", 1, 1, churn=10), one_row("t1", 2, 1, churn=20),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "churn_fail_ratio"] == pytest.approx(0.0)


def test_metadata_carried_from_first_row():
    hist = make_history([
        one_row("t1", 1, 1, assertions=7, age=120),
        one_row("t1", 2, 0, assertions=7, age=120),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "num_assertions"] == 7
    assert feats.loc[0, "age_days"] == 120


def test_fail_churn_corr_positive_when_churn_drives_failures():
    # failures happen exactly on the two high-churn runs -> positive corr
    hist = make_history([
        one_row("t1", 1, 1, churn=1), one_row("t1", 2, 1, churn=1),
        one_row("t1", 3, 0, churn=100), one_row("t1", 4, 0, churn=100),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "fail_churn_corr"] > 0.9


def test_fail_churn_corr_zero_when_no_variance():
    hist = make_history([
        one_row("t1", 1, 1, churn=5), one_row("t1", 2, 1, churn=5),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "fail_churn_corr"] == pytest.approx(0.0)


def test_multiple_tests_aggregated_independently():
    hist = make_history([
        one_row("t1", 1, 1), one_row("t1", 2, 1),
        one_row("t2", 1, 0), one_row("t2", 2, 0),
    ])
    feats = build_features(hist)
    assert len(feats) == 2
    by_id = feats.set_index("test_id")
    assert by_id.loc["t1", "fail_rate"] == pytest.approx(0.0)
    assert by_id.loc["t2", "fail_rate"] == pytest.approx(1.0)


def test_unsorted_runs_are_sorted_by_run_id():
    # rows deliberately out of order: pass,pass,fail -> sorted -> 0.25 flips
    hist = make_history([
        one_row("t1", 3, 0), one_row("t1", 1, 1), one_row("t1", 2, 1),
    ])
    feats = build_features(hist)
    assert feats.loc[0, "flip_rate"] == pytest.approx(0.5)
    assert feats.loc[0, "fail_rate"] == pytest.approx(1 / 3)
