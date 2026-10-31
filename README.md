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
    └── metrics.csv             metrics of every model on test_data.csv
```

## d. Models used

All five models are trained on the same training split and evaluated on the same
held-out `test_data.csv`.

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000, C=1.0)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=12, min_samples_leaf=3)` |
| KNN | `KNeighborsClassifier(n_neighbors=7, weights="distance")` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | RBF-kernel `SVC(C=2.0)` wrapped in `CalibratedClassifierCV` for probability scores |

### Comparison table (test set, 5,039 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9748 | 0.9956 | 0.9805 | 0.9651 | 0.9728 | 0.9494 |
| Decision Tree | 0.9958 | 0.9979 | 0.9966 | 0.9945 | 0.9955 | 0.9916 |
| KNN | 0.9946 | 0.9980 | 0.9957 | 0.9928 | 0.9942 | 0.9892 |
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

* **CSV upload** of test data (sidebar) – plus a button to download the sample `test_data.csv`.
* **Model selection dropdown** for the five saved classifiers.
* **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1, MCC for the chosen model.
* **Confusion matrix and classification report** for the chosen model.
* **Compare all models** tab – metric table, bar chart for any metric, and ROC curves.
* Download of the predictions of the selected model.

The app only loads the saved `.joblib` models; it never retrains on the uploaded data.

## How to run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py      # optional – re-creates model/*.joblib and test_data.csv
streamlit run app.py
```
