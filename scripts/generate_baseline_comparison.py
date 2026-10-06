"""
Phase 2 — Baseline model comparison tables.

This script consolidates previously generated Phase 2 results into
publication-ready comparison tables.

Sources:

    - Hyperparameter tuning results
    - Validation metrics
    - Held-out test metrics
    - Error analysis summary
    - Computational-cost analysis

No model training or retraining is performed.
"""

from __future__ import annotations

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


TUNING_FILE = (
    OUTPUT_DIR
    / "hyperparameter_tuning_results.json"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "validation_metrics.json"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test_metrics.json"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "error_analysis_summary.csv"
)

COST_FILE = (
    OUTPUT_DIR
    / "computational_cost_analysis.json"
)


CSV_OUTPUT = (
    OUTPUT_DIR
    / "baseline_model_comparison.csv"
)

JSON_OUTPUT = (
    OUTPUT_DIR
    / "baseline_model_comparison.json"
)

MARKDOWN_OUTPUT = (
    OUTPUT_DIR
    / "baseline_model_comparison.md"
)


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )

    return pd.read_csv(path)


def get_best_validation_results(
    tuning_data: dict,
) -> dict:
    """
    Extract validation results for the selected models.

    Model selection is based on the already completed tuning experiment.
    """

    selected = {}

    for modality in ["email", "url"]:

        modality_data = tuning_data.get(
            modality,
            {},
        )

        for model_name, model_data in modality_data.items():

            best_result = model_data.get(
                "best_result"
            )

            if not best_result:
                continue

            selected[model_name] = {
                "modality": modality,
                "parameters": best_result.get(
                    "parameters",
                    {},
                ),
                "validation_accuracy": best_result.get(
                    "accuracy"
                ),
                "validation_precision": best_result.get(
                    "precision"
                ),
                "validation_recall": best_result.get(
                    "recall"
                ),
                "validation_f1": best_result.get(
                    "f1"
                ),
                "validation_roc_auc": best_result.get(
                    "roc_auc"
                ),
                "validation_pr_auc": best_result.get(
                    "pr_auc"
                ),
            }

    return selected


def extract_test_results(
    test_data: dict,
) -> dict:
    """
    Extract held-out test metrics.

    The script supports the current Phase 2 JSON structure.
    """

    results = {}

    for result in test_data.get(
        "models",
        [],
    ):

        model_name = result.get(
            "model"
        )

        if not model_name:
            continue

        results[model_name] = result

    return results


def extract_error_results(
    error_df: pd.DataFrame,
) -> dict:
    """Convert error-analysis rows into a model-indexed dictionary."""

    results = {}

    for _, row in error_df.iterrows():

        model_name = row.get(
            "Model"
        )

        if pd.isna(model_name):
            continue

        results[str(model_name)] = {
            "test_samples": row.get(
                "Test Samples"
            ),
            "false_positives": row.get(
                "False Positives"
            ),
            "false_negatives": row.get(
                "False Negatives"
            ),
            "false_positive_rate": row.get(
                "False Positive Rate"
            ),
            "false_negative_rate": row.get(
                "False Negative Rate"
            ),
        }

    return results


def extract_cost_results(
    cost_data: dict,
) -> dict:
    """Convert computational-cost results into a model-indexed dictionary."""

    results = {}

    for result in cost_data.get(
        "models",
        [],
    ):

        model_name = result.get(
            "model"
        )

        if not model_name:
            continue

        results[model_name] = result

    return results


def build_comparison_table(
    validation_results: dict,
    test_results: dict,
    error_results: dict,
    cost_results: dict,
) -> pd.DataFrame:
    """Build the consolidated baseline comparison table."""

    rows = []

    for model_name in validation_results:

        validation = validation_results[
            model_name
        ]

        test = test_results.get(
            model_name,
            {},
        )

        error = error_results.get(
            model_name,
            {},
        )

        cost = cost_results.get(
            model_name,
            {},
        )

        row = {
            "Modality": validation.get(
                "modality"
            ),

            "Model": model_name,

            "Parameters": json.dumps(
                validation.get(
                    "parameters",
                    {},
                ),
                sort_keys=True,
            ),

            "Validation Accuracy": validation.get(
                "validation_accuracy"
            ),

            "Validation Precision": validation.get(
                "validation_precision"
            ),

            "Validation Recall": validation.get(
                "validation_recall"
            ),

            "Validation F1": validation.get(
                "validation_f1"
            ),

            "Validation ROC-AUC": validation.get(
                "validation_roc_auc"
            ),

            "Validation PR-AUC": validation.get(
                "validation_pr_auc"
            ),

            "Test Accuracy": test.get(
                "accuracy"
            ),

            "Test Precision": test.get(
                "precision"
            ),

            "Test Recall": test.get(
                "recall"
            ),

            "Test F1": test.get(
                "f1"
            ),

            "Test ROC-AUC": test.get(
                "roc_auc"
            ),

            "Test PR-AUC": test.get(
                "pr_auc"
            ),

            "Test Samples": error.get(
                "test_samples"
            ),

            "False Positives": error.get(
                "false_positives"
            ),

            "False Negatives": error.get(
                "false_negatives"
            ),

            "False Positive Rate": error.get(
                "false_positive_rate"
            ),

            "False Negative Rate": error.get(
                "false_negative_rate"
            ),

            "Features": cost.get(
                "num_features"
            ),

            "Transformation Time (s)": cost.get(
                "feature_transformation_time_seconds"
            ),

            "Prediction Time (s)": cost.get(
                "prediction_time_seconds"
            ),

            "Probability Time (s)": cost.get(
                "probability_time_seconds"
            ),

            "Total Inference Time (s)": cost.get(
                "total_inference_time_seconds"
            ),

            "Time per Sample (ms)": cost.get(
                "total_time_per_sample_ms"
            ),

            "Throughput (samples/s)": cost.get(
                "throughput_samples_per_second"
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def round_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Round floating-point columns for readable output."""

    dataframe = dataframe.copy()

    for column in dataframe.columns:

        if pd.api.types.is_float_dtype(
            dataframe[column]
        ):

            dataframe[column] = dataframe[
                column
            ].round(6)

    return dataframe


def generate_markdown_table(
    dataframe: pd.DataFrame,
) -> str:
    """
    Generate a compact Markdown table.

    The full CSV/JSON outputs retain all fields.
    """

    columns = [
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
    ]

    table = dataframe[
        columns
    ].copy()

    return table.to_markdown(
        index=False,
        floatfmt=".4f",
    )


def main() -> None:
    """Generate Phase 2 baseline comparison tables."""

    print("=" * 80)
    print(
        "PHASE 2 — BASELINE MODEL COMPARISON"
    )
    print("=" * 80)

    print("\nLoading existing experiment results...")

    tuning_data = load_json(
        TUNING_FILE
    )

    test_data = load_json(
        TEST_FILE
    )

    error_df = load_csv(
        ERROR_FILE
    )

    cost_data = load_json(
        COST_FILE
    )

    print("  Hyperparameter tuning results : loaded")
    print("  Held-out test results         : loaded")
    print("  Error analysis                : loaded")
    print("  Computational cost analysis   : loaded")

    validation_results = (
        get_best_validation_results(
            tuning_data
        )
    )

    test_results = extract_test_results(
        test_data
    )

    error_results = extract_error_results(
        error_df
    )

    cost_results = extract_cost_results(
        cost_data
    )

    comparison = build_comparison_table(
        validation_results=validation_results,
        test_results=test_results,
        error_results=error_results,
        cost_results=cost_results,
    )

    if comparison.empty:
        raise RuntimeError(
            "The comparison table is empty. "
            "Check the Phase 2 result files."
        )

    comparison = round_numeric_columns(
        comparison
    )

    comparison = comparison.sort_values(
        by=[
            "Modality",
            "Test F1",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    comparison.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------------
    # JSON
    # ---------------------------------------------------------------

    json_records = comparison.to_dict(
        orient="records"
    )

    json_output = {
        "experiment": (
            "phase2_baseline_model_comparison"
        ),
        "retraining_performed": False,
        "models": json_records,
    }

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_output,
            file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Markdown
    # ---------------------------------------------------------------

    markdown_table = generate_markdown_table(
        comparison
    )

    markdown_content = (
        "# Phase 2 Baseline Model Comparison\n\n"
        "This table consolidates the Phase 2 baseline "
        "validation, held-out test, error-analysis, and "
        "computational-cost results.\n\n"
        "**No model retraining was performed during this "
        "comparison step.**\n\n"
        + markdown_table
        + "\n"
    )

    MARKDOWN_OUTPUT.write_text(
        markdown_content,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Console output
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison[
            [
                "Modality",
                "Model",
                "Validation F1",
                "Test Accuracy",
                "Test Precision",
                "Test Recall",
                "Test F1",
                "Test ROC-AUC",
                "Test PR-AUC",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(CSV_OUTPUT)
    print(JSON_OUTPUT)
    print(MARKDOWN_OUTPUT)

    print(
        "\nSTEP 22 BASELINE COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()