"""
Tests for Phase 2 baseline model comparison artifacts.
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)


def test_baseline_comparison_files_exist():
    csv_file = (
        OUTPUT_DIR
        / "baseline_model_comparison.csv"
    )

    json_file = (
        OUTPUT_DIR
        / "baseline_model_comparison.json"
    )

    markdown_file = (
        OUTPUT_DIR
        / "baseline_model_comparison.md"
    )

    assert csv_file.exists()
    assert json_file.exists()
    assert markdown_file.exists()


def test_baseline_comparison_contains_expected_models():
    csv_file = (
        OUTPUT_DIR
        / "baseline_model_comparison.csv"
    )

    dataframe = pd.read_csv(csv_file)

    models = set(
        dataframe["Model"].tolist()
    )

    expected_models = {
        "email_logistic_regression",
        "email_linear_svm",
        "email_random_forest",
        "url_logistic_regression",
        "url_random_forest",
        "url_gradient_boosting",
    }

    assert expected_models.issubset(models)


def test_baseline_comparison_contains_required_metrics():
    csv_file = (
        OUTPUT_DIR
        / "baseline_model_comparison.csv"
    )

    dataframe = pd.read_csv(csv_file)

    required_columns = {
        "Modality",
        "Model",
        "Validation F1",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Test ROC-AUC",
        "Test PR-AUC",
        "False Positives",
        "False Negatives",
        "Time per Sample (ms)",
        "Throughput (samples/s)",
    }

    assert required_columns.issubset(
        dataframe.columns
    )


def test_baseline_comparison_json_schema():
    json_file = (
        OUTPUT_DIR
        / "baseline_model_comparison.json"
    )

    with json_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert (
        data["experiment"]
        == "phase2_baseline_model_comparison"
    )

    assert (
        data["retraining_performed"]
        is False
    )

    assert "models" in data

    assert len(
        data["models"]
    ) >= 6