"""
Tests for Step 13 targeted hyperparameter tuning.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "tune_baseline_models_fast.py"
)


def test_step13_script_exists():
    assert SCRIPT_PATH.exists()


def test_step13_script_has_expected_models():

    content = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "email_logistic_regression" in content
    assert "email_linear_svm" in content
    assert "email_random_forest" in content

    assert "url_logistic_regression" in content
    assert "url_random_forest" in content
    assert "url_gradient_boosting" in content


def test_step13_uses_validation_data():

    content = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "validation.csv" in content
    assert "validation" in content


def test_step13_does_not_load_test_data():

    content = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "test.csv" not in content


def test_step13_records_partial_email_rf():

    content = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "partial_completed" in content
    assert "0.9835" in content
    assert "max_features" in content