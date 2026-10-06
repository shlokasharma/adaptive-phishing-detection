"""
Phase 2 — Automated baseline artifact validation.

This script validates the presence and basic structural integrity
of the finalized Phase 2 baseline artifacts.

The validator does NOT:

    - retrain models
    - modify model artifacts
    - modify experiment results
    - perform hyperparameter tuning

It only checks that the expected Phase 2 outputs exist and that
their basic schemas are internally consistent.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase2_baseline.yaml"
)


REQUIRED_MODEL_ARTIFACTS = [
    "email_logistic_regression.pkl",
    "email_linear_svm.pkl",
    "email_random_forest.pkl",
    "url_logistic_regression.pkl",
    "url_random_forest.pkl",
    "url_gradient_boosting.pkl",
]


REQUIRED_EXTRACTOR_ARTIFACTS = [
    "email_logistic_regression_tfidf_extractor.pkl",
    "email_linear_svm_tfidf_extractor.pkl",
    "email_random_forest_tfidf_extractor.pkl",
    "url_logistic_regression_feature_extractor.pkl",
    "url_random_forest_feature_extractor.pkl",
    "url_gradient_boosting_feature_extractor.pkl",
]


REQUIRED_OUTPUT_FILES = [
    "hyperparameter_tuning_results.json",
    "validation_metrics.json",
    "test_metrics.json",
    "error_analysis_summary.csv",
    "model_confidence_analysis.json",
    "model_confidence_summary.csv",
    "computational_cost_analysis.json",
    "computational_cost_summary.csv",
    "baseline_model_comparison.csv",
    "baseline_model_comparison.json",
    "baseline_model_comparison.md",
]


def check_file_exists(
    path: Path,
) -> tuple[bool, str]:
    """Check whether a required file exists."""

    if path.exists():
        return True, f"PASS: {path.name}"

    return False, f"MISSING: {path}"


def validate_model_artifacts() -> list[str]:
    """Validate required model and extractor artifacts."""

    messages = []

    for filename in REQUIRED_MODEL_ARTIFACTS:

        path = MODEL_DIR / filename

        exists, message = check_file_exists(
            path
        )

        if not exists:
            raise FileNotFoundError(
                message
            )

        messages.append(message)

        with path.open("rb") as file:
            model = pickle.load(file)

        if not hasattr(
            model,
            "predict",
        ):
            raise TypeError(
                f"Model artifact does not expose "
                f"predict(): {filename}"
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                f"Model artifact does not expose "
                f"predict_proba(): {filename}"
            )

    for filename in REQUIRED_EXTRACTOR_ARTIFACTS:

        path = MODEL_DIR / filename

        exists, message = check_file_exists(
            path
        )

        if not exists:
            raise FileNotFoundError(
                message
            )

        messages.append(message)

        with path.open("rb") as file:
            extractor = pickle.load(file)

        if not hasattr(
            extractor,
            "transform",
        ):
            raise TypeError(
                f"Feature extractor artifact does not "
                f"expose transform(): {filename}"
            )

    return messages


def validate_output_files() -> list[str]:
    """Validate required Phase 2 output files."""

    messages = []

    for filename in REQUIRED_OUTPUT_FILES:

        path = OUTPUT_DIR / filename

        exists, message = check_file_exists(
            path
        )

        if not exists:
            raise FileNotFoundError(
                message
            )

        messages.append(message)

    return messages


def validate_configuration() -> str:
    """Validate the Phase 2 configuration file."""

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 2 configuration:\n"
            f"{CONFIG_FILE}"
        )

    content = CONFIG_FILE.read_text(
        encoding="utf-8"
    )

    required_terms = [
        "phase2_baseline_experiments",
        "random_state: 42",
        "test_set_used_for_model_selection: false",
        "email_linear_svm",
        "url_random_forest",
    ]

    for term in required_terms:

        if term not in content:
            raise ValueError(
                f"Phase 2 configuration is missing "
                f"required term: {term}"
            )

    return "PASS: Phase 2 configuration"


def validate_comparison_table() -> dict:
    """Validate the consolidated baseline comparison table."""

    csv_path = (
        OUTPUT_DIR
        / "baseline_model_comparison.csv"
    )

    dataframe = pd.read_csv(
        csv_path
    )

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

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Baseline comparison table is missing "
            f"columns: {sorted(missing_columns)}"
        )

    expected_models = {
        "email_logistic_regression",
        "email_linear_svm",
        "email_random_forest",
        "url_logistic_regression",
        "url_random_forest",
        "url_gradient_boosting",
    }

    actual_models = set(
        dataframe["Model"].tolist()
    )

    missing_models = (
        expected_models
        - actual_models
    )

    if missing_models:
        raise ValueError(
            "Baseline comparison table is missing "
            f"models: {sorted(missing_models)}"
        )

    return {
        "rows": int(len(dataframe)),
        "models": sorted(actual_models),
    }


def validate_test_metrics() -> dict:
    """Validate held-out test metrics."""

    path = (
        OUTPUT_DIR
        / "test_metrics.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    required_modalities = {
        "email",
        "url",
    }

    available_modalities = {
        key
        for key in data.keys()
        if key in required_modalities
    }

    missing_modalities = (
        required_modalities
        - available_modalities
    )

    if missing_modalities:
        raise ValueError(
            "test_metrics.json is missing required "
            f"modalities: {sorted(missing_modalities)}"
        )

    # ---------------------------------------------------------------
    # Validate email selected baseline
    # ---------------------------------------------------------------

    email_result = data["email"]

    if email_result.get(
        "model_name"
    ) != "email_linear_svm":
        raise ValueError(
            "Unexpected selected email model: "
            f"{email_result.get('model_name')}"
        )

    required_email_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "test_samples",
    }

    missing_email_metrics = (
        required_email_metrics
        - set(email_result.keys())
    )

    if missing_email_metrics:
        raise ValueError(
            "Email test metrics are missing fields: "
            f"{sorted(missing_email_metrics)}"
        )

    # ---------------------------------------------------------------
    # Validate URL selected baseline
    # ---------------------------------------------------------------

    url_result = data["url"]

    if url_result.get(
        "model_name"
    ) != "url_random_forest":
        raise ValueError(
            "Unexpected selected URL model: "
            f"{url_result.get('model_name')}"
        )

    required_url_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "test_samples",
    }

    missing_url_metrics = (
        required_url_metrics
        - set(url_result.keys())
    )

    if missing_url_metrics:
        raise ValueError(
            "URL test metrics are missing fields: "
            f"{sorted(missing_url_metrics)}"
        )

    return {
        "modalities": sorted(
            available_modalities
        ),
        "email_model": email_result[
            "model_name"
        ],
        "url_model": url_result[
            "model_name"
        ],
        "email_test_samples": email_result[
            "test_samples"
        ],
        "url_test_samples": url_result[
            "test_samples"
        ],
    }


def validate_error_analysis() -> dict:
    """Validate error-analysis results."""

    path = (
        OUTPUT_DIR
        / "error_analysis_summary.csv"
    )

    dataframe = pd.read_csv(
        path
    )

    required_columns = {
        "Modality",
        "Model",
        "Test Samples",
        "False Positives",
        "False Negatives",
        "False Positive Rate",
        "False Negative Rate",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Error-analysis summary is missing "
            f"columns: {sorted(missing_columns)}"
        )

    return {
        "rows": int(len(dataframe)),
    }


def main() -> None:
    """Run all Phase 2 artifact validations."""

    print("=" * 80)
    print(
        "PHASE 2 — BASELINE ARTIFACT VALIDATION"
    )
    print("=" * 80)

    validation_results = {
        "experiment": (
            "phase2_baseline_artifact_validation"
        ),
        "status": "PASS",
        "checks": {},
    }

    print("\n[1/6] Validating model artifacts...")

    model_messages = validate_model_artifacts()

    validation_results["checks"][
        "model_artifacts"
    ] = {
        "status": "PASS",
        "details": model_messages,
    }

    print(
        f"  ✓ {len(model_messages)} model/extractor "
        f"artifacts validated"
    )

    print("\n[2/6] Validating output files...")

    output_messages = validate_output_files()

    validation_results["checks"][
        "output_files"
    ] = {
        "status": "PASS",
        "details": output_messages,
    }

    print(
        f"  ✓ {len(output_messages)} output files validated"
    )

    print("\n[3/6] Validating configuration...")

    configuration_message = (
        validate_configuration()
    )

    validation_results["checks"][
        "configuration"
    ] = {
        "status": "PASS",
        "details": configuration_message,
    }

    print(
        f"  ✓ {configuration_message}"
    )

    print("\n[4/6] Validating comparison table...")

    comparison_result = (
        validate_comparison_table()
    )

    validation_results["checks"][
        "comparison_table"
    ] = {
        "status": "PASS",
        "details": comparison_result,
    }

    print(
        f"  ✓ {comparison_result['rows']} comparison rows"
    )

    print("\n[5/6] Validating test metrics...")

    test_result = validate_test_metrics()

    validation_results["checks"][
        "test_metrics"
    ] = {
        "status": "PASS",
        "details": test_result,
    }

    print(
        "  ✓ Selected models present in test results"
    )

    print("\n[6/6] Validating error analysis...")

    error_result = validate_error_analysis()

    validation_results["checks"][
        "error_analysis"
    ] = {
        "status": "PASS",
        "details": error_result,
    }

    print(
        f"  ✓ {error_result['rows']} error-analysis rows"
    )

    validation_path = (
        OUTPUT_DIR
        / "phase2_artifact_validation.json"
    )

    with validation_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            validation_results,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(
        "PHASE 2 ARTIFACT VALIDATION COMPLETE"
    )
    print("=" * 80)

    print(
        "\nStatus: PASS"
    )

    print(
        "\nValidation report:"
    )

    print(validation_path)


if __name__ == "__main__":
    main()