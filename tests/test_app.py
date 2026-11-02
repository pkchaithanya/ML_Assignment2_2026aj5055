"""
End-to-end tests for the Streamlit app, driven through Streamlit's AppTest
(the real app.py runs; nothing is mocked).

Run from the project root:
    pip install -r requirements-dev.txt
    python -m pytest
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
TEST_CSV = ROOT / "test_data.csv"
METRICS_CSV = ROOT / "model" / "metrics.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
TIMEOUT = 120  # scoring five models (SVM is the slowest) on the full test file


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_CSV)


def run_app():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def upload(at, df, name="upload.csv"):
    """Upload a DataFrame as CSV through the app's file uploader and rerun."""
    at.file_uploader[0].upload(name, df.to_csv(index=False).encode(), "text/csv")
    return at.run()


def upload_bytes(at, content, name="upload.csv"):
    at.file_uploader[0].upload(name, content, "text/csv")
    return at.run()


def shown_metrics(at):
    """Evaluation metric cards currently on screen, as {label: value string}."""
    return {m.label: m.value for m in at.metric if m.label in METRIC_NAMES}


def assert_no_exception(at):
    assert not at.exception, [e.value for e in at.exception]


# ---------------- Before upload ----------------
def test_empty_state_shows_guidance_but_no_results():
    at = run_app()
    assert_no_exception(at)
    assert not at.metric, "metrics must only appear after a file is uploaded"
    assert any(s.value == "How it works" for s in at.subheader)
    assert not at.error


# ---------------- Valid uploads ----------------
@pytest.fixture(scope="module")
def full_upload(test_df):
    """App with the full test_data.csv uploaded (shared: scoring all models takes a few seconds)."""
    return upload(run_app(), test_df, "test_data.csv")


def test_valid_upload_shows_dataset_summary(full_upload, test_df):
    at = full_upload
    assert_no_exception(at)
    assert not at.error
    summary = {m.label: m.value for m in at.metric}
    n_anomaly = int((test_df["class"] == "anomaly").sum())
    assert summary["Rows"] == f"{len(test_df):,}"
    assert summary["Anomalies"] == f"{n_anomaly:,}"
    assert summary["Normal"] == f"{len(test_df) - n_anomaly:,}"


@pytest.mark.parametrize("model", pd.read_csv(METRICS_CSV, index_col=0).index.tolist())
def test_each_model_reproduces_saved_test_metrics(full_upload, model):
    """The app must score the saved models to exactly the numbers recorded at training time."""
    at = full_upload
    model_box = next(s for s in at.selectbox if s.label == "Model")
    model_box.select(model).run()
    assert_no_exception(at)

    expected = pd.read_csv(METRICS_CSV, index_col=0).loc[model]
    shown = shown_metrics(at)
    assert set(shown) == set(METRIC_NAMES)
    for name in METRIC_NAMES:
        assert shown[name] == f"{expected[name]:.4f}", f"{model} {name}"


def test_numeric_target_is_accepted(test_df):
    """A 0/1 target column gives the same results as the 'normal'/'anomaly' labels."""
    sample = test_df.sample(300, random_state=1)
    as_text = shown_metrics(upload(run_app(), sample))
    numeric = sample.assign(**{"class": (sample["class"] == "anomaly").astype(int)})
    at = upload(run_app(), numeric)
    assert_no_exception(at)
    assert not at.error
    assert shown_metrics(at) == as_text


def test_extra_and_reordered_columns_are_ignored(test_df):
    sample = test_df.sample(300, random_state=2)
    baseline = shown_metrics(upload(run_app(), sample))
    shuffled = sample[list(reversed(sample.columns))].assign(note="extra column")
    at = upload(run_app(), shuffled)
    assert_no_exception(at)
    assert shown_metrics(at) == baseline


@pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")  # expected for one class
def test_single_class_file_reports_auc_as_na(test_df):
    """AUC is undefined with one class; the app must say so instead of crashing."""
    only_normal = test_df[test_df["class"] == "normal"].head(200)
    at = upload(run_app(), only_normal)
    assert_no_exception(at)
    assert shown_metrics(at)["AUC"] == "n/a"
    assert any("ROC curves need both classes" in c.value for c in at.caption)


# ---------------- Invalid uploads ----------------
def test_missing_columns_are_reported(test_df):
    dropped = ["src_bytes", "service"]
    at = upload(run_app(), test_df.head(50).drop(columns=dropped))
    assert_no_exception(at)
    assert len(at.error) == 1
    assert "missing 2 required column(s)" in at.error[0].value
    assert any(all(f"`{c}`" in md.value for c in dropped) for md in at.markdown)
    assert not shown_metrics(at), "no results may be shown for an invalid file"


def test_missing_target_column_is_reported(test_df):
    at = upload(run_app(), test_df.head(50).drop(columns=["class"]))
    assert_no_exception(at)
    assert "missing 1 required column(s)" in at.error[0].value
    assert not shown_metrics(at)


def test_unexpected_target_values_are_reported(test_df):
    bad = test_df.head(50).copy()
    bad.loc[bad.index[0], "class"] = "attack"
    at = upload(run_app(), bad)
    assert_no_exception(at)
    assert "Unexpected target values: ['attack']" in at.error[0].value
    assert not shown_metrics(at)


def test_unreadable_csv_is_reported():
    at = upload_bytes(run_app(), b"")
    assert_no_exception(at)
    assert at.error and "Could not read the CSV" in at.error[0].value
    assert not shown_metrics(at)
