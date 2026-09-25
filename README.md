# Flaky Test Predictor

I spent three years as a Software Test Engineer watching CI pipelines go red for
no reason — a test fails, you re-run it, it passes, nobody knows why. Flaky
tests erode trust in the whole suite: engineers start ignoring failures, and
real regressions slip through. So I built an ML model that predicts which tests
are flaky from their run history, so teams can quarantine or fix them before
they poison the pipeline.

## Approach
1. **Synthetic CI history** (`src/generate_data.py`): 300 tests × 60 runs, 13%
   flaky. Flaky tests fail intermittently regardless of code churn and have
   unstable durations; stable tests fail only on high churn (real regressions).
   A spectrum of borderline cases keeps the problem honest.
2. **Feature engineering** (`src/features.py`): per-test aggregates — failure
   rate, flip rate (pass→fail→pass oscillation), duration mean/std/CV, and two
   churn-interaction features (`churn_fail_ratio`, `fail_churn_corr`) that capture
   whether failures are *explained by* code changes. That last idea comes straight
   from triaging real failures: a test that fails on quiet days is suspicious.
3. **Model**: RandomForestClassifier (300 trees, balanced class weights).

## Results (5-fold CV on the synthetic data)
| Metric (flaky class) | Score |
|---|---|
| Precision | 0.82 ± 0.12 |
| Recall | 0.83 ± 0.10 |
| F1 | 0.81 ± 0.05 |

Top features by importance: duration CV (0.44), duration std, flip rate,
failure rate — matching the intuition that *timing instability* is the
strongest flakiness signal. The misses are borderline cases (mildly flaky vs.
mildly unstable), which is exactly where a human would struggle too.

## Project structure
```
├── src/
│   ├── generate_data.py   # synthetic CI history + ground-truth labels
│   ├── features.py        # per-test feature engineering
│   ├── train.py           # train, cross-validate, save model
│   └── score.py           # rank a new suite by P(flaky)
├── requirements.txt
└── README.md
```
`data/` (generated history) and `models/` (trained model) are created by the
scripts and git-ignored.

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py   # create data/test_history.csv + data/labels.csv
python src/train.py           # train + 5-fold CV + save models/flaky_model.pkl
python src/score.py           # rank tests: python src/score.py [history_csv]
```

## Design decisions
- **Why these features:** I chose features a QA engineer would actually look at
  when triaging — not just "failure rate" but *why* it failed (churn context)
  and *how* it behaves (timing stability).
- **Why RandomForest:** tabular features, small data, need feature importances
  to explain predictions to a skeptical QA lead. No deep learning needed.
- **Honest evaluation:** stratified 5-fold CV instead of a single lucky split,
  and the synthetic data includes genuinely ambiguous cases.
