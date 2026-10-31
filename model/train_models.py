"""
Train the five classifiers required by ML Assignment 2 on the Kaggle
"Network Intrusion Detection" dataset and save them under model/.

Usage (from the project root):
    python model/train_models.py

Outputs
-------
model/<name>.joblib    one fitted Pipeline (preprocessing + classifier) per model
model/feature_info.json  feature lists / label mapping used by the Streamlit app
model/metrics.csv      evaluation metrics of every model on the held-out test set
test_data.csv          held-out 20% test split (features + target), for the app
"""

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
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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


def main():
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
    rows = []
    for name, clf in get_classifiers().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric_cols)), ("clf", clf)])
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


if __name__ == "__main__":
    main()
