"""
Tests for Phase 2 baseline training artifacts.

These tests verify that the expected trained model artifacts exist
after the baseline training scripts have been executed.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
)


EXPECTED_ARTIFACTS = [
    "email_logistic_regression.pkl",
    "email_linear_svm.pkl",
    "email_random_forest.pkl",
    "email_logistic_regression_tfidf_extractor.pkl",
    "email_linear_svm_tfidf_extractor.pkl",
    "email_random_forest_tfidf_extractor.pkl",
    "url_logistic_regression.pkl",
    "url_random_forest.pkl",
    "url_gradient_boosting.pkl",
    "url_logistic_regression_feature_extractor.pkl",
    "url_random_forest_feature_extractor.pkl",
    "url_gradient_boosting_feature_extractor.pkl",
]


def test_baseline_model_directory_exists():
    assert MODEL_DIR.exists()
    assert MODEL_DIR.is_dir()


def test_all_expected_baseline_artifacts_exist():
    missing_artifacts = [
        artifact
        for artifact in EXPECTED_ARTIFACTS
        if not (MODEL_DIR / artifact).exists()
    ]

    assert not missing_artifacts, (
        "Missing baseline artifacts: "
        f"{missing_artifacts}"
    )


def test_expected_artifact_count():
    pkl_files = list(
        MODEL_DIR.glob("*.pkl")
    )

    assert len(pkl_files) >= len(
        EXPECTED_ARTIFACTS
    )