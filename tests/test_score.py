"""Unit tests for src/score.py — ranking a test suite by predicted flakiness.

score.py is a thin CLI: it reads a run-history CSV, engineers features with
src/features.py, loads the pickled model bundle, and prints the tests ranked
by P(flaky). Tests train a tiny RandomForest on hand-built histories (a
clearly stable test vs a clearly flaky one) so the expected ranking is
checkable by reading the test.
"""
import os
import pickle
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

# repo root on sys.path so `from src.features import ...` resolves, and src/
# on sys.path so score.py's own `from features import ...` resolves
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import score  # noqa: E402
from src.features import FEATURE_COLS, build_features  # noqa: E402

COLS = ["test_id", "run_id", "passed", "duration_s", "churn_lines",
        "num_assertions", "age_days"]


def make_history(rows):
    """rows: list of (test_id, run_id, passed, duration_s, churn_lines,
    num_assertions, age_days)"""
    return pd.DataFrame(rows, columns=COLS)


def stable_runs(test_id, n=8, duration=10.0):
    """A rock-stable test: always passes, steady timing, low churn."""
    return [(test_id, i, 1, duration, 5, 10, 100) for i in range(n)]


def flaky_runs(test_id, n=8):
    """A textbook flaky test: oscillates pass/fail, timing all over the place."""
    return [(test_id, i, 1 if i % 2 == 0 else 0, 5.0 + 5.0 * (i % 4), 5, 10, 100)
            for i in range(n)]


def train_tiny_model(tmp_path):
    """Train a small RandomForest on 4 stable + 4 flaky hand-built tests and
    save the score.py bundle (model + feature list) to tmp_path."""
    train_rows = []
    for i in range(4):
        train_rows += stable_runs(f"stable_{i}", duration=10.0 + i)
        train_rows += flaky_runs(f"flaky_{i}")
    feats = build_features(make_history(train_rows))
    labels = [0 if tid.startswith("stable") else 1 for tid in feats["test_id"]]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(feats[FEATURE_COLS].to_numpy(), labels)
    model_path = tmp_path / "flaky_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURE_COLS}, f)
    return str(model_path)


def write_history(tmp_path, rows):
    csv_path = tmp_path / "history.csv"
    make_history(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def run_score(monkeypatch, capsys, csv_path, model_path):
    monkeypatch.setattr(score, "MODEL_PATH", model_path)
    monkeypatch.setattr(sys, "argv", ["score.py", csv_path])
    score.main()
    return capsys.readouterr().out


def test_ranking_puts_flaky_first(tmp_path, monkeypatch, capsys):
    model_path = train_tiny_model(tmp_path)
    csv_path = write_history(tmp_path, stable_runs("t_stable") + flaky_runs("t_flaky"))
    out = run_score(monkeypatch, capsys, csv_path, model_path)
    assert "scored 2 tests" in out
    # the flaky test must rank above the stable one in the printed table
    assert out.index("t_flaky") < out.index("t_stable")


def test_flaky_flagged_and_stable_not(tmp_path, monkeypatch, capsys):
    model_path = train_tiny_model(tmp_path)
    csv_path = write_history(tmp_path, stable_runs("t_stable") + flaky_runs("t_flaky"))
    out = run_score(monkeypatch, capsys, csv_path, model_path)
    assert "LIKELY FLAKY" in out
    assert "stable" in out
    assert "1 of 2 tests flagged as likely flaky" in out


def test_extra_history_columns_are_ignored(tmp_path, monkeypatch, capsys):
    # real histories carry columns score.py never uses (e.g. suite); they
    # must not break scoring
    model_path = train_tiny_model(tmp_path)
    hist = make_history(stable_runs("t_stable") + flaky_runs("t_flaky"))
    hist["suite"] = "unit"
    csv_path = tmp_path / "history.csv"
    hist.to_csv(csv_path, index=False)
    out = run_score(monkeypatch, capsys, str(csv_path), model_path)
    assert "scored 2 tests" in out


def test_single_test_suite_scores(tmp_path, monkeypatch, capsys):
    model_path = train_tiny_model(tmp_path)
    csv_path = write_history(tmp_path, flaky_runs("t_only"))
    out = run_score(monkeypatch, capsys, csv_path, model_path)
    assert "scored 1 tests" in out
    assert "t_only" in out


def test_missing_model_exits_with_helpful_message(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "MODEL_PATH", str(tmp_path / "no_such_model.pkl"))
    monkeypatch.setattr(sys, "argv", ["score.py", str(tmp_path / "history.csv")])
    with pytest.raises(SystemExit, match="train the model first"):
        score.main()


def test_default_paths_point_at_repo_data_and_model():
    assert score.DEFAULT_HISTORY.endswith(os.path.join("data", "test_history.csv"))
    assert score.MODEL_PATH.endswith(os.path.join("models", "flaky_model.pkl"))
