"""
Train the five classifiers required by ML Assignment 2 on the Kaggle
"Network Intrusion Detection" dataset and save them under model/.

Usage (from the project root):
    python model/train_models.py                  # tune, train, cross-validate and evaluate
    python model/train_models.py --skip-cv        # faster: skip the 5-fold cross-validation
    python model/train_models.py --skip-tuning    # use the default hyperparameters as-is

Outputs
-------
model/<name>.joblib    one fitted Pipeline (preprocessing + classifier) per model
model/feature_info.json  feature lists / label mapping used by the Streamlit app
model/metrics.csv      evaluation metrics of every model on the held-out test set
model/cv_scores.csv    5-fold cross-validation mean / std of every metric (training split only)
model/tuning_results.json  grid-search results for the tuned models (KNN, Decision Tree)
model/feature_importances.csv  Decision Tree importance of each original dataset column
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
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

# Hyperparameter grids searched before the final fit ("clf__" targets the classifier step of
# the pipeline). MCC is the selection score because it uses all four confusion-matrix cells.
PARAM_GRIDS = {
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
    },
    "Decision Tree": {
        "clf__max_depth": [6, 8, 10, 12, 16, 20, None],
        "clf__min_samples_leaf": [1, 2, 3, 5],
        "clf__criterion": ["gini", "entropy"],
    },
}


def load_dataset():
    df = pd.read_csv(RAW_DATA)
    # Duplicate rows could land in both the train and test split and inflate the test score.
    df = df.drop_duplicates().reset_index(drop=True)

    # Columns with a single value carry no information (e.g. num_outbound_cmds).
    constant_cols = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    return df, constant_cols


def build_preprocessor(numeric_cols):
    # Living inside each model's Pipeline means the scaler and encoder are fitted on the
    # training rows only, and the saved .joblib file can take raw CSV rows in the app.
    return ColumnTransformer(
        transformers=[
            # Byte counts reach millions while rates sit in 0-1; scaling stops the large
            # columns dominating the distance (KNN, SVM) and coefficient (Logistic) based models.
            ("num", StandardScaler(), numeric_cols),
            # dense output so GaussianNB can consume it; unseen categories are ignored
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def get_classifiers():
    # Starting settings; KNN and the Decision Tree are replaced by the grid-search winner
    # (see PARAM_GRIDS) unless --skip-tuning is passed.
    return {
        # max_iter raised from the default 100 so the solver converges on ~120 encoded columns
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        # depth / leaf limits keep the tree from memorising individual training rows
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=SEED),
        # distance weighting lets the closest neighbours count more than the far ones
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


def tune_model(pipe, grid, X_train, y_train):
    """Grid search on the training split; returns the refit best pipeline and a summary."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, cv=folds, scoring=CV_SCORING["MCC"], n_jobs=-1)
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    top = [
        {"params": {k.removeprefix("clf__"): v for k, v in r["params"].items()},
         "mean_mcc": round(float(r["mean_test_score"]), 4),
         "std_mcc": round(float(r["std_test_score"]), 4)}
        for _, r in results.head(5).iterrows()
    ]
    summary = {
        "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        "best_cv_mcc": round(float(search.best_score_), 4),
        "combinations_tried": len(results),
        "top_5": top,
    }
    return search.best_estimator_, summary


def feature_importances(pipe):
    """Decision Tree importances mapped back to the original dataset columns.

    One-hot columns such as "cat__service_http" are summed into their source column
    ("service"); for categorical columns the single most important value is kept as well.
    """
    encoded_names = pipe.named_steps["prep"].get_feature_names_out()
    weights = pipe.named_steps["clf"].feature_importances_

    rows = []
    for encoded, weight in zip(encoded_names, weights):
        group, name = encoded.split("__", 1)
        if group == "cat":
            column = next(c for c in CATEGORICAL if name.startswith(c + "_"))
            rows.append((column, "categorical", name[len(column) + 1:], weight))
        else:
            rows.append((name, "numeric", None, weight))
    parts = pd.DataFrame(rows, columns=["feature", "type", "value", "importance"])

    table = parts.groupby(["feature", "type"], as_index=False)["importance"].sum()
    top_value = (parts.dropna(subset=["value"]).sort_values("importance", ascending=False)
                 .drop_duplicates("feature").set_index("feature")["value"])
    table["top_value"] = table["feature"].map(top_value)
    return table.sort_values("importance", ascending=False).round(4).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the five classifiers.")
    parser.add_argument("--skip-cv", action="store_true", help="skip the 5-fold cross-validation")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="keep the default hyperparameters instead of grid-searching them")
    args = parser.parse_args()

    # ---- Step 1: load and clean the Kaggle data ------------------------------------------
    df, dropped = load_dataset()
    print(f"Dataset after cleaning: {df.shape[0]} rows, {df.shape[1] - 1} features "
          f"(dropped constant columns: {dropped})")
    print("Class balance:\n", df[TARGET].value_counts().to_string(), "\n")

    # ---- Step 2: features / target, then an 80/20 train-test split ------------------------
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)  # anomaly = 1, normal = 0
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]

    # stratify keeps the ~53/47 normal/anomaly ratio identical in both parts, and the
    # fixed random_state makes the split (and every score below) reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # Keep the held-out split with its original text labels so the app can be fed it as-is.
    test_df = X_test.copy()
    test_df[TARGET] = np.where(y_test == 1, POSITIVE_LABEL, "normal")
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved test split ({len(test_df)} rows) -> {TEST_CSV.name}")

    # ---- Step 3: for each model, tune -> cross-validate -> fit -> evaluate -> save -------
    # Everything up to the final fit only ever sees X_train; X_test is touched once, at the
    # end, so the test metrics are an honest estimate for unseen traffic.
    MODEL_DIR.mkdir(exist_ok=True)
    rows, cv_rows, tuning = [], [], {}
    for name, clf in get_classifiers().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric_cols)), ("clf", clf)])

        # 3a. hyperparameter search (KNN and Decision Tree only)
        if name in PARAM_GRIDS and not args.skip_tuning:
            start = time.perf_counter()
            pipe, tuning[name] = tune_model(pipe, PARAM_GRIDS[name], X_train, y_train)
            print(f"{name:<24} tuned in {time.perf_counter() - start:6.1f}s  "
                  f"best {tuning[name]['best_params']}  CV MCC={tuning[name]['best_cv_mcc']:.4f} "
                  f"({tuning[name]['combinations_tried']} combinations)")
        # Note: for tuned models the CV below reuses the folds the grid search chose on,
        # so it is slightly optimistic; the held-out test set gives the unbiased check.

        # 3b. 5-fold cross-validation: shows how stable the score is across different splits
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

        # 3c. final fit on the whole training split
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        # 3d. evaluate on the held-out test split; y_score (probability of "anomaly") is
        # needed for AUC, the hard 0/1 predictions for the other five metrics
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        scores = evaluate(y_test, y_pred, y_score)
        rows.append({"Model": name, **scores})

        # 3e. save the whole pipeline (preprocessing + model) for the Streamlit app
        joblib.dump(pipe, MODEL_DIR / file_name(name), compress=3)
        print(f"{name:<24} trained in {elapsed:6.1f}s  "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

        if name == "Decision Tree":
            importances = feature_importances(pipe)
            importances.to_csv(MODEL_DIR / "feature_importances.csv", index=False)

    # ---- Step 4: write the result files the app and README use ----------------------------
    metrics = pd.DataFrame(rows).set_index("Model").round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv")

    # the app reads this to know which columns to expect and which file holds which model
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
    if tuning:
        (MODEL_DIR / "tuning_results.json").write_text(json.dumps(tuning, indent=2))

    # ---- Step 5: console summary ------------------------------------------------------------
    print("\nTop 10 Decision Tree features (importance, summed over one-hot columns):")
    print(importances.head(10).fillna("-").to_string(index=False))

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
