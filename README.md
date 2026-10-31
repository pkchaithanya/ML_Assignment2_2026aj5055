# Network Intrusion Detection – ML Assignment 2

Live app: `<add your Streamlit Community Cloud link here>`

## a. Problem statement

Computer networks are constantly probed and attacked. The goal of this project is to
build a binary classifier that looks at the summary features of a single network
connection (protocol, service, bytes transferred, error rates, host statistics, …)
and decides whether the connection is **normal** traffic or an **anomaly**
(an intrusion / attack). Five classical machine-learning models are trained on the
same data, compared on six evaluation metrics, and served through an interactive
Streamlit web application.

## b. Dataset description

| Item | Value |
|---|---|
| Source | Kaggle – *Network Intrusion Detection* (sampadab17), derived from the KDD Cup '99 / NSL-KDD intrusion data. https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection |
| File used | `Train_data.csv` (the Kaggle `Test_data.csv` has no `class` label, so it cannot be used to compute metrics) |
| Instances | 25,192 connections (no missing values, no duplicate rows) |
| Features | 41 input features (38 numeric, 3 categorical: `protocol_type`, `service`, `flag`) |
| Target | `class` – `normal` (13,449 rows, 53.4%) / `anomaly` (11,743 rows, 46.6%) |
| Task | Binary classification (`anomaly` = positive class) |

**Preprocessing** (all of it is part of each saved scikit-learn `Pipeline`, so the app
can be given raw rows):

* Two constant columns (`num_outbound_cmds`, `is_host_login`) are dropped → **39 features used**.
* Numeric features are standardised with `StandardScaler`.
* Categorical features are one-hot encoded (`handle_unknown="ignore"` so an unseen
  service/flag in uploaded data does not crash the app).
* Stratified 80 / 20 train–test split (`random_state=26`): 20,153 training rows,
  5,039 test rows. The 20 % test split is saved as `test_data.csv`.

## c. GitHub repository link

`<add your GitHub repository link here>`

```
ml-assignment-2/
│-- app.py                      Streamlit application
│-- .streamlit/config.toml      theme (light + dark)
│-- assets/style.css            responsive card grids
│-- requirements.txt
│-- README.md
│-- test_data.csv               held-out test split (features + class) – upload this in the app
│-- data/
│   └── Train_data.csv          original Kaggle data
└── model/
    ├── train_models.py         training + evaluation script
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── feature_info.json       feature lists / label mapping used by the app
    ├── metrics.csv             metrics of every model on test_data.csv
    ├── cv_scores.csv           5-fold cross-validation mean / std on the training split
    └── tuning_results.json     grid-search results for KNN and Decision Tree
```

## d. Models used

All five models are trained on the same training split and evaluated on the same
held-out `test_data.csv`.

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000, C=1.0)` |
| Decision Tree | `DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=1)` – tuned |
| KNN | `KNeighborsClassifier(n_neighbors=3, weights="distance")` – tuned |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | RBF-kernel `SVC(C=2.0)` wrapped in `CalibratedClassifierCV` for probability scores |

### Hyperparameter tuning

KNN and the Decision Tree were tuned with `GridSearchCV` (stratified 5-fold CV on the
training split only, selecting by MCC). Full results are in `model/tuning_results.json`.

| Model | Grid searched | Best setting | CV MCC |
|---|---|---|---|
| Decision Tree | `max_depth` ∈ {6, 8, 10, 12, 16, 20, None}, `min_samples_leaf` ∈ {1, 2, 3, 5}, `criterion` ∈ {gini, entropy} (56 combinations) | entropy, depth 10, leaf 1 | 0.9932 |
| KNN | `n_neighbors` ∈ {3, 5, 7, 9, 11, 15}, `weights` ∈ {uniform, distance} (12 combinations) | k = 3, distance | 0.9894 |

Effect on the test set compared with the earlier hand-picked settings:

| Model | MCC before → after | F1 before → after | AUC before → after |
|---|---|---|---|
| Decision Tree (depth 12, leaf 3 → tuned) | 0.9916 → 0.9948 | 0.9955 → 0.9972 | 0.9979 → 0.9974 |
| KNN (k = 7 → tuned) | 0.9892 → 0.9904 | 0.9942 → 0.9949 | 0.9980 → 0.9973 |

### Comparison table (test set, 5,039 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9748 | 0.9956 | 0.9805 | 0.9651 | 0.9728 | 0.9494 |
| Decision Tree | 0.9974 | 0.9974 | 0.9987 | 0.9957 | 0.9972 | 0.9948 |
| KNN | 0.9952 | 0.9973 | 0.9962 | 0.9936 | 0.9949 | 0.9904 |
| Naive Bayes | 0.8625 | 0.9855 | 0.9976 | 0.7067 | 0.8273 | 0.7480 |
| Support Vector Machine | 0.9921 | 0.9991 | 0.9927 | 0.9902 | 0.9915 | 0.9841 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | `<write your own observation>` |
| Decision Tree | `<write your own observation>` |
| KNN | `<write your own observation>` |
| Naive Bayes | `<write your own observation>` |
| Support Vector Machine | `<write your own observation>` |
| Overall, winner for this dataset? | `<write your own conclusion>` |

## Streamlit app features

* **CSV upload** of test data at the top of the page – plus a button to download the sample `test_data.csv`.
* **Model selection dropdown** for the five saved classifiers.
* **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1, MCC for the chosen model (Results tab).
* **Confusion matrix and classification report** for the chosen model.
* **Compare** tab – metric table with the best score per column highlighted, bar chart for any metric, and ROC curves.
* **Features** tab – which dataset columns drive the Decision Tree's predictions
  (from `model/feature_importances.csv`), with an adjustable top-N chart and the full table.
* **Data** tab – preview of the uploaded file, and a download of the selected model's predictions.
* Clear error messages for missing columns or unexpected target values.
* Responsive layout (phone, tablet, desktop) and light/dark theme, configured in
  `.streamlit/config.toml` with a small stylesheet in `assets/style.css`.

The app only loads the saved `.joblib` models; it never retrains on the uploaded data.
Metrics are computed once per uploaded file and cached, so switching models is instant.

## How to run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py      # optional – re-creates model/*.joblib and test_data.csv
streamlit run app.py
```
