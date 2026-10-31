"""
Train the five classifiers required by ML Assignment 2 on the Kaggle
"Network Intrusion Detection" dataset and save them under model/.

Usage (from the project root):
    python model/train_models.py              # train, cross-validate and evaluate
    python model/train_models.py --skip-cv    # faster: skip the 5-fold cross-validation

Outputs
-------
model/<name>.joblib    one fitted Pipeline (preprocessing + classifier) per model
model/feature_info.json  feature lists / label mapping used by the Streamlit app
model/metrics.csv      evaluation metrics of every model on the held-out test set
model/cv_scores.csv    5-fold cross-validation mean / std of every metric (training split only)
test_data.csv          held-out 20% test split (features + target), for the app
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "Train_data.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "class"
POSITIVE_LABEL = "anomaly"  # an intrusion is the positive class (1)
CATEGORICAL = ["protocol_type", "service", "flag"]
SEED = 26
CV_FOLDS = 5

# scikit-learn scorer names for the six assignment metrics
CV_SCORING = {
    "Accuracy": "accuracy",
    "AUC": "roc_auc",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "MCC": make_scorer(matthews_corrcoef),
}


def load_dataset():
    df = pd.read_csv(RAW_DATA)
    df = df.drop_duplicates().reset_index(drop=True)

    # Columns with a single value carry no information (e.g. num_outbound_cmds).
    constant_cols = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    return df, constant_cols


def build_preprocessor(numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            # dense output so GaussianNB can consume it; unseen categories are ignored
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        # Platt-scaled SVC so predict_proba is available for the AUC score
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", C=2.0, gamma="scale"), ensemble=False
        ),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def cross_validate_model(pipe, X_train, y_train):
    """Stratified k-fold CV on the training split only, so test_data.csv stays unseen.

    Returns {metric: (mean, std)} across the folds.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv = cross_validate(pipe, X_train, y_train, cv=folds, scoring=CV_SCORING, n_jobs=-1)
    return {m: (cv[f"test_{m}"].mean(), cv[f"test_{m}"].std()) for m in CV_SCORING}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the five classifiers.")
    parser.add_argument("--skip-cv", action="store_true", help="skip the 5-fold cross-validation")
    args = parser.parse_args()

    df, dropped = load_dataset()
    print(f"Dataset after cleaning: {df.shape[0]} rows, {df.shape[1] - 1} features "
          f"(dropped constant columns: {dropped})")
    print("Class balance:\n", df[TARGET].value_counts().to_string(), "\n")

    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # Keep the held-out split with its original text labels so the app can be fed it as-is.
    test_df = X_test.copy()
    test_df[TARGET] = np.where(y_test == 1, POSITIVE_LABEL, "normal")
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved test split ({len(test_df)} rows) -> {TEST_CSV.name}")

    MODEL_DIR.mkdir(exist_ok=True)
    rows, cv_rows = [], []
    for name, clf in get_classifiers().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric_cols)), ("clf", clf)])

        if not args.skip_cv:
            start = time.perf_counter()
            cv_scores = cross_validate_model(pipe, X_train, y_train)
            row = {"Model": name}
            for m, (mean, std) in cv_scores.items():
                row[f"{m} mean"], row[f"{m} std"] = mean, std
            cv_rows.append(row)
            # "+/-" instead of the plus-minus sign so it prints on any Windows console
            print(f"{name:<24} {CV_FOLDS}-fold CV in {time.perf_counter() - start:6.1f}s  "
                  + "  ".join(f"{m}={mean:.4f}+/-{std:.4f}" for m, (mean, std) in cv_scores.items()))

        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        scores = evaluate(y_test, y_pred, y_score)
        rows.append({"Model": name, **scores})

        joblib.dump(pipe, MODEL_DIR / file_name(name), compress=3)
        print(f"{name:<24} trained in {elapsed:6.1f}s  "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = pd.DataFrame(rows).set_index("Model").round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv")

    info = {
        "target": TARGET,
        "positive_label": POSITIVE_LABEL,
        "negative_label": "normal",
        "categorical_features": CATEGORICAL,
        "numeric_features": numeric_cols,
        "dropped_constant_columns": dropped,
        "models": {name: file_name(name) for name in get_classifiers()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": SEED,
    }
    (MODEL_DIR / "feature_info.json").write_text(json.dumps(info, indent=2))

    print("\nComparison table (test set):")
    print(metrics.to_string())

    if cv_rows:
        cv_table = pd.DataFrame(cv_rows).set_index("Model").round(4)
        cv_table.to_csv(MODEL_DIR / "cv_scores.csv")
        summary = pd.DataFrame(
            {m: [f"{r[f'{m} mean']:.4f} +/- {r[f'{m} std']:.4f}" for _, r in cv_table.iterrows()]
             for m in CV_SCORING},
            index=cv_table.index,
        )
        print(f"\n{CV_FOLDS}-fold cross-validation on the training split (mean +/- std):")
        print(summary.to_string())


if __name__ == "__main__":
    main()
