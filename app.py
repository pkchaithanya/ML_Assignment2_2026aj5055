"""
Streamlit front-end for ML Assignment 2 - Network Intrusion Detection.

The app only *loads* the models saved by model/train_models.py; nothing is
retrained on the uploaded data.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
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
# translucent tint + bold keeps the theme's own text colour readable in light and dark mode
BEST_STYLE = "background-color: rgba(45, 212, 191, 0.28); font-weight: 700"
METRIC_HELP = {
    "Accuracy": "Share of all rows classified correctly.",
    "AUC": "Area under the ROC curve: how well the model ranks anomalies above normal traffic.",
    "Precision": "Of the rows flagged as anomaly, the share that really are anomalies.",
    "Recall": "Of the real anomalies, the share the model caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient: balanced score from -1 to 1 using all four confusion-matrix cells.",
}

st.set_page_config(page_title="Intrusion Detection - Model Lab", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    info = json.loads((MODEL_DIR / "feature_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in info["models"].items()}
    return info, models


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


@st.cache_data(show_spinner="Scoring the five models on your data…")
def score_all(X, y, _models):
    """Score every saved model once per uploaded file; switching models is then instant."""
    results = {}
    for name, model in _models.items():
        metrics, y_pred, y_prob = score_model(model, X, y)
        results[name] = {"metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}
    return results


# ---------------- Charts (Altair follows the app theme and resizes with its container) ----------------
def confusion_chart(cm, labels):
    data = pd.DataFrame(
        [(labels[i], labels[j], int(cm[i, j])) for i in range(2) for j in range(2)],
        columns=["Actual", "Predicted", "Count"],
    )
    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=labels),
    )
    cells = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=["Actual", "Predicted", "Count"],
    )
    text = base.mark_text(fontSize=20, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(alt.datum.Count > data["Count"].max() / 2,
                            alt.value("white"), alt.value("#0f172a")),
    )
    return (cells + text).properties(height=280)


def metric_bar_chart(table, metric):
    data = table[[metric]].reset_index()
    # bars start at a zoomed baseline instead of 0 so small differences stay visible
    data["base"] = max(0.0, float(data[metric].min()) - 0.05)
    base = alt.Chart(data).encode(
        y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[data["base"].iloc[0], 1.0], zero=False),
                title=metric),
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x2="base:Q",
        color=alt.Color("Model:N", legend=None, sort=list(table.index)),
        tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
    )
    # neutral grey just past the bar end reads on both themes, even for very short bars
    labels = base.mark_text(align="left", dx=4, color="#8b95a5", fontWeight="bold").encode(
        text=alt.Text(f"{metric}:Q", format=".4f")
    )
    return (bars + labels).properties(height=260)


def roc_chart(y, results):
    frames = []
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["y_prob"])
        auc = res["metrics"]["AUC"]
        frames.append(pd.DataFrame({"FPR": fpr, "TPR": tpr, "Model": name,
                                    "Legend": f"{name} ({auc:.3f})"}))
    data = pd.concat(frames)
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("FPR:Q", title="False positive rate"),
        y=alt.Y("TPR:Q", title="True positive rate"),
        color=alt.Color("Legend:N", sort=[f for f in data["Legend"].unique()],
                        # the lower-right of a ROC plot is empty, so the legend sits inside it
                        legend=alt.Legend(orient="bottom-right", direction="vertical", labelLimit=0,
                                          title="Model (AUC)", padding=8, cornerRadius=6,
                                          fillColor="rgba(128,128,128,0.12)")),
        tooltip=["Model", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
    )
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
    return (chance + lines).properties(height=320)


def report_table(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=class_names,
                                   output_dict=True, zero_division=0)
    rows = class_names + ["macro avg", "weighted avg"]
    table = pd.DataFrame(report).T.loc[rows]
    table["support"] = table["support"].astype(int)
    # short headers keep the table narrow enough for a phone screen
    table = table.rename(columns={"precision": "Precision", "recall": "Recall",
                                  "f1-score": "F1", "support": "Support"},
                         index={"macro avg": "Macro avg", "weighted avg": "Weighted"})
    return table.style.format("{:.4f}", subset=["Precision", "Recall", "F1"])


def metric_cards(items, fmt="{:.4f}"):
    """Six bordered metric cards in a responsive grid (6 / 3 / 2 per row, see assets/style.css)."""
    with st.container(key="cq_metrics"), st.container(key="grid_metrics"):
        for label, value in items:
            shown = "n/a" if isinstance(value, float) and np.isnan(value) else fmt.format(value)
            st.metric(label, shown, help=METRIC_HELP.get(label), border=True)


st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)
info, models = load_artifacts()
feature_cols = info["numeric_features"] + info["categorical_features"]
class_names = [info["negative_label"], info["positive_label"]]
target = info["target"]

# ---------------- Sidebar: background info only, the app works without opening it ----------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "Five classifiers trained on 80% of the Kaggle **Network Intrusion Detection** "
        "dataset predict whether a network connection is `normal` or an `anomaly`."
    )
    st.markdown("**Models**\n\n" + "\n".join(f"- {m}" for m in models))
    st.caption("Uploaded data is only used for prediction and scoring. "
               "The models are never retrained in the app.")

# ---------------- Header + controls ----------------
st.title("🛡️ Intrusion Detection Classifier Lab")
st.caption("Upload labelled network-connection records and see how five trained models "
           "separate **normal** traffic from **anomalous** (intrusion) traffic.")

with st.container(border=True):
    col_upload, col_model = st.columns([3, 2], gap="medium")
    with col_upload:
        uploaded = st.file_uploader(
            "Test data (CSV)", type=["csv"],
            help=f"Must contain the {len(feature_cols)} feature columns and the '{target}' column.",
        )
    with col_model:
        model_name = st.selectbox("Model", list(models.keys()))
        if SAMPLE_CSV.exists():
            st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv",
                               icon=":material/download:", width="stretch")

# ---------------- Empty state ----------------
if uploaded is None:
    st.subheader("How it works")
    steps = [
        (":material/download:", "1. Get test data",
         "Use your own labelled CSV or download the sample test_data.csv above."),
        (":material/upload_file:", "2. Upload it",
         "Drop the file into the uploader. Nothing is retrained."),
        (":material/insights:", "3. Review results",
         "Metrics, confusion matrix and report for the chosen model, plus a side-by-side comparison."),
    ]
    for col, (icon, title, text) in zip(st.columns(3), steps):
        with col.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)

    with st.expander(f"Required columns ({len(feature_cols) + 1})"):
        st.markdown(" ".join(f"`{c}`" for c in feature_cols + [target]))
        st.caption(f"`{target}` must be `{class_names[0]}` / `{class_names[1]}` (or 0 / 1).")
    st.stop()

# ---------------- Validate upload ----------------
try:
    df = pd.read_csv(uploaded)
except Exception as exc:  # malformed upload
    st.error(f"Could not read the CSV: {exc}", icon=":material/error:")
    st.stop()

missing = [c for c in feature_cols + [target] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s).",
             icon=":material/error:")
    with st.expander("Show missing columns", expanded=len(missing) <= 10):
        st.markdown(" ".join(f"`{c}`" for c in missing))
    st.stop()

try:
    y_true = encode_target(df[target], info)
except ValueError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()
X = df[feature_cols]
results = score_all(X, y_true, models)

# ---------------- Dataset summary ----------------
n_rows, n_anom = len(df), int(y_true.sum())
with st.container(key="cq_summary"), st.container(key="grid_summary"):
    st.metric("Rows", f"{n_rows:,}", border=True, height="stretch")
    st.metric("Normal", f"{n_rows - n_anom:,}", delta=f"{(n_rows - n_anom) / n_rows:.1%}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Anomalies", f"{n_anom:,}", delta=f"{n_anom / n_rows:.1%}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")

tab_single, tab_compare, tab_data = st.tabs(
    [":material/analytics: Results", ":material/leaderboard: Compare", ":material/table_view: Data"]
)

with tab_single:
    res = results[model_name]
    y_pred = res["y_pred"]

    st.subheader(model_name)
    metric_cards(res["metrics"].items())

    left, right = st.columns([1, 1.25], gap="medium")
    with left.container(border=True):
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        st.altair_chart(confusion_chart(cm, class_names), width="stretch")
    with right.container(border=True):
        st.markdown("**Classification report**")
        st.table(report_table(y_true, y_pred, class_names))
        result = df.copy()
        result["predicted"] = np.where(y_pred == 1, info["positive_label"], info["negative_label"])
        st.download_button("Download predictions", result.to_csv(index=False).encode(),
                           file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                           mime="text/csv", icon=":material/download:", type="primary")

with tab_compare:
    table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
    table.index.name = "Model"

    best = table["MCC"].idxmax()
    st.markdown(f"Best on this data by MCC: :green-badge[:material/trophy: {best}]")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, props=BEST_STYLE),
                 width="stretch")
    st.caption("Highlighted cells are the best score in each column.")

    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        st.markdown("**Metric by model**")
        metric_pick = st.selectbox("Metric", list(table.columns), index=list(table.columns).index("MCC"))
        st.altair_chart(metric_bar_chart(table, metric_pick), width="stretch")
    with right.container(border=True):
        st.markdown("**ROC curves**")
        if len(np.unique(y_true)) > 1:
            st.altair_chart(roc_chart(y_true, results), width="stretch")
        else:
            st.caption("ROC curves need both classes in the uploaded data.")

with tab_data:
    st.caption(f"First 100 of {n_rows:,} rows · {df.shape[1]} columns")
    st.dataframe(df.head(100), width="stretch", height=420)
