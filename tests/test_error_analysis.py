"""
Tests for Phase 2 error-analysis artifacts.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)


def _check_error_report(filename: str) -> None:
    path = OUTPUT_DIR / filename

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_fields = {
        "experiment",
        "model",
        "test_samples",
        "legitimate_samples",
        "phishing_samples",
        "false_positive_count",
        "false_negative_count",
        "false_positive_rate",
        "false_negative_rate",
    }

    assert required_fields.issubset(data.keys())

    assert data["test_samples"] >= 0
    assert data["false_positive_count"] >= 0
    assert data["false_negative_count"] >= 0

    assert 0.0 <= data["false_positive_rate"] <= 1.0
    assert 0.0 <= data["false_negative_rate"] <= 1.0


def test_email_error_analysis_schema():
    _check_error_report(
        "email_error_analysis.json"
    )


def test_url_error_analysis_schema():
    _check_error_report(
        "url_error_analysis.json"
    )


def test_error_analysis_counts_are_valid():
    for filename in [
        "email_error_analysis.json",
        "url_error_analysis.json",
    ]:
        path = OUTPUT_DIR / filename

        if not path.exists():
            continue

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        assert (
            data["false_positive_count"]
            + data["false_negative_count"]
            <= data["test_samples"]
        )