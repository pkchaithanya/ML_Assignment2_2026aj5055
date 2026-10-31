"""
Streamlit front-end for ML Assignment 2 - Network Intrusion Detection.

The app only *loads* the models saved by model/train_models.py; nothing is
retrained on the uploaded data.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
# readable in both light and dark themes
BEST_STYLE = "background-color: #2e7d32; color: white; font-weight: 600"

st.set_page_config(page_title="Intrusion Detection - Model Lab", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    info = json.loads((MODEL_DIR / "feature_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in info["models"].items()}
    return info, models


@st.cache_data
def load_reference_metrics():
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


def encode_target(series, info):
    """Map the text target to 0/1. Accepts 'normal'/'anomaly' or already-numeric 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    labels = series.astype(str).str.strip().str.lower()
    unknown = set(labels.unique()) - {info["positive_label"], info["negative_label"]}
    if unknown:
        raise ValueError(f"Unexpected target values: {sorted(unknown)}")
    return (labels == info["positive_label"]).astype(int).to_numpy()


def score_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        # AUC is undefined when the uploaded file contains a single class
        "AUC": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_prob


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc(y, scores_by_model):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, prob in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


info, models = load_artifacts()
feature_cols = info["numeric_features"] + info["categorical_features"]
class_names = [info["negative_label"], info["positive_label"]]

# ---------------- Sidebar ----------------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("CSV with all features + 'class' column", type=["csv"])
if SAMPLE_CSV.exists():
    st.sidebar.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Classifier", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models are pre-trained on 80% of the Kaggle *Network Intrusion Detection* data. "
    "Uploaded data is only used for prediction and scoring."
)

# ---------------- Main ----------------
st.title("🛡️ Network Intrusion Detection – Classifier Lab")
st.write(
    "Upload labelled network-connection records to see how each of the five trained "
    "classifiers separates **normal** traffic from **anomalous** (intrusion) traffic."
)

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin. "
            "You can download the sample `test_data.csv` there if you need one.")
    ref = load_reference_metrics()
    if ref is not None:
        st.subheader("Reference metrics on the held-out test split")
        st.dataframe(ref.style.format("{:.4f}").highlight_max(axis=0, props=BEST_STYLE),
                     width="stretch")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:  # malformed upload
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

target = info["target"]
missing = [c for c in feature_cols + [target] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {missing}")
    st.stop()

try:
    y_true = encode_target(df[target], info)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
X = df[feature_cols]

c1, c2, c3 = st.columns(3)
c1.metric("Rows uploaded", f"{len(df):,}")
c2.metric("Anomalies", f"{int(y_true.sum()):,}")
c3.metric("Normal", f"{int((1 - y_true).sum()):,}")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(50), width="stretch")

tab_single, tab_compare = st.tabs([f"📊 {model_name}", "🏁 Compare all models"])

with tab_single:
    metrics, y_pred, _ = score_model(models[model_name], X, y_true)

    st.subheader("Evaluation metrics")
    items = list(metrics.items())
    for row in (items[:3], items[3:]):
        for col, (k, v) in zip(st.columns(3), row):
            col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        st.pyplot(plot_confusion(cm, class_names))
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=[0, 1], target_names=class_names,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch")

    result = df.copy()
    result["predicted"] = np.where(y_pred == 1, info["positive_label"], info["negative_label"])
    st.download_button("Download predictions", result.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

with tab_compare:
    rows, probs = [], {}
    for name, model in models.items():
        m, _, prob = score_model(model, X, y_true)
        rows.append({"Model": name, **m})
        probs[name] = prob
    table = pd.DataFrame(rows).set_index("Model")

    st.subheader("Metrics of every model on the uploaded data")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, props=BEST_STYLE),
                 width="stretch")

    left, right = st.columns(2)
    with left:
        metric_pick = st.selectbox("Metric to compare", list(table.columns), index=5)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        sns.barplot(x=table[metric_pick].values, y=table.index, ax=ax, palette="viridis",
                    hue=table.index, legend=False)
        ax.set_xlabel(metric_pick)
        ax.set_ylabel("")
        ax.set_xlim(max(0, table[metric_pick].min() - 0.05), 1.0)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        if len(np.unique(y_true)) > 1:
            st.pyplot(plot_roc(y_true, probs))
        else:
            st.caption("ROC curves need both classes in the uploaded data.")
