"""Train the flaky-test classifier and report honest metrics.

Model: RandomForestClassifier on the engineered per-test features.
Evaluation: stratified 80/20 split, precision / recall / F1 on the flaky class,
plus confusion matrix and feature importances. The trained model is saved to
models/flaky_model.pkl for score.py.
"""
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import FEATURE_COLS, load_labeled

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models")


def make_model():
    return RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)


def main():
    df = load_labeled()
    X = df[FEATURE_COLS].to_numpy()
    y = df["is_flaky"].to_numpy()

    # 5-fold cross-validated metrics (stable estimate on the small flaky class)
    precisions, recalls, f1s = [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        clf = make_model().fit(X[tr], y[tr])
        p, r, f, _ = precision_recall_fscore_support(
            y[te], clf.predict(X[te]), average="binary", zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    import numpy as np
    print(f"{len(df)} tests ({int(y.sum())} flaky), 5-fold CV:")
    print(f"precision (flaky): {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    print(f"recall    (flaky): {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"F1        (flaky): {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = make_model()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0)
    print(f"test set: {len(y_test)} tests "
          f"({int(y_test.sum())} flaky)")
    print(f"precision (flaky): {precision:.3f}")
    print(f"recall    (flaky): {recall:.3f}")
    print(f"F1        (flaky): {f1:.3f}")
    print("\nconfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))
    print("\nclassification report:")
    print(classification_report(y_test, pred, target_names=["stable", "flaky"],
                                zero_division=0))

    importances = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    print("feature importances:")
    for name, imp in importances:
        print(f"  {name:18s} {imp:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "flaky_model.pkl"), "wb") as f:
        pickle.dump({"model": clf, "features": FEATURE_COLS}, f)
    print(f"\nmodel saved -> {MODEL_DIR}/flaky_model.pkl")


if __name__ == "__main__":
    main()
