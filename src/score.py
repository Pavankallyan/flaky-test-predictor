"""Score a new test suite: rank tests by predicted flakiness probability.

Usage:
    python src/score.py [history_csv]

Reads a run-history CSV with the same columns as data/test_history.csv
(no labels needed), engineers the features, loads models/flaky_model.pkl,
and prints the tests ranked by P(flaky), highest first.
"""
import os
import pickle
import sys

import pandas as pd

from features import build_features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_HISTORY = os.path.join(ROOT, "data", "test_history.csv")
MODEL_PATH = os.path.join(ROOT, "models", "flaky_model.pkl")


def main():
    history_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HISTORY
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("train the model first: python src/train.py")
    history = pd.read_csv(history_csv)
    feats = build_features(history)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    proba = bundle["model"].predict_proba(feats[bundle["features"]].to_numpy())[:, 1]

    ranked = feats[["test_id"]].copy()
    ranked["p_flaky"] = proba
    ranked["verdict"] = ["LIKELY FLAKY - quarantine/review"
                         if p >= 0.5 else "stable" for p in proba]
    ranked = ranked.sort_values("p_flaky", ascending=False)

    print(f"scored {len(ranked)} tests from {history_csv}\n")
    print(ranked.head(15).to_string(index=False))
    n_flagged = int((ranked["p_flaky"] >= 0.5).sum())
    print(f"\n{n_flagged} of {len(ranked)} tests flagged as likely flaky")


if __name__ == "__main__":
    main()
