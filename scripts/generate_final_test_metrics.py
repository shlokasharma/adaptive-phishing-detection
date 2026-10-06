"""
Generate the final Phase 2 held-out test metric summary.

This script reads the selected-model test evaluation results and
produces publication-friendly JSON and CSV summaries.

Selected models:
    Email -> Linear SVM
    URL   -> Random Forest

Metrics:
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC
    PR-AUC
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
    / "test_metrics.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

OUTPUT_JSON = OUTPUT_DIR / "final_test_metrics.json"
OUTPUT_CSV = OUTPUT_DIR / "final_test_metrics.csv"


REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


def load_test_metrics() -> dict:
    """Load the held-out test evaluation results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Test metrics file not found:\n{INPUT_FILE}\n\n"
            "Run Step 15 first to generate test_metrics.json."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_model_metrics(data: dict) -> list[dict]:
    """
    Extract the selected email and URL model metrics.

    The function is intentionally tolerant of the surrounding JSON
    structure so that the reporting step does not depend on unnecessary
    metadata fields.
    """

    results: list[dict] = []

    def search_models(obj: object, modality: str | None = None) -> None:
        if isinstance(obj, dict):

            # Detect dictionaries containing classification metrics.
            if all(metric in obj for metric in REQUIRED_METRICS):

                model_name = (
                    obj.get("model_name")
                    or obj.get("model")
                    or obj.get("name")
                    or "unknown_model"
                )

                current_modality = (
                    obj.get("modality")
                    or modality
                    or "unknown"
                )

                row = {
                    "modality": current_modality,
                    "model": model_name,
                }

                for metric in REQUIRED_METRICS:
                    row[metric] = float(obj[metric])

                if "confusion_matrix" in obj:
                    row["confusion_matrix"] = obj["confusion_matrix"]

                results.append(row)

            for key, value in obj.items():
                child_modality = modality

                key_lower = str(key).lower()

                if "email" in key_lower:
                    child_modality = "email"
                elif "url" in key_lower:
                    child_modality = "url"

                search_models(value, child_modality)

        elif isinstance(obj, list):
            for item in obj:
                search_models(item, modality)

    search_models(data)

    return results


def deduplicate_results(results: list[dict]) -> list[dict]:
    """Remove duplicate metric records."""

    unique: dict[tuple[str, str], dict] = {}

    for result in results:
        key = (
            str(result["modality"]).lower(),
            str(result["model"]).lower(),
        )

        unique[key] = result

    return list(unique.values())


def normalize_model_names(results: list[dict]) -> list[dict]:
    """Normalize model names for the final report."""

    normalized = []

    for result in results:
        row = dict(result)

        model_name = str(row["model"]).lower()

        if "linear_svm" in model_name:
            row["model"] = "Linear SVM"
        elif "random_forest" in model_name:
            row["model"] = "Random Forest"
        elif "logistic_regression" in model_name:
            row["model"] = "Logistic Regression"
        elif "gradient_boosting" in model_name:
            row["model"] = "Gradient Boosting"

        normalized.append(row)

    return normalized


def select_final_models(results: list[dict]) -> list[dict]:
    """
    Keep the final selected Phase 2 models.

    Email:
        Linear SVM

    URL:
        Random Forest
    """

    selected = []

    for result in results:
        modality = str(result["modality"]).lower()
        model = str(result["model"]).lower()

        if modality == "email" and "linear svm" in model:
            selected.append(result)

        elif modality == "url" and "random forest" in model:
            selected.append(result)

    return selected


def create_summary_table(results: list[dict]) -> pd.DataFrame:
    """Create the publication-friendly summary table."""

    rows = []

    for result in results:
        rows.append(
            {
                "Modality": result["modality"].title(),
                "Model": result["model"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "ROC-AUC": result["roc_auc"],
                "PR-AUC": result["pr_auc"],
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["Modality"],
            ascending=True,
        ).reset_index(drop=True)

    return df


def main() -> None:
    """Run final test metric reporting."""

    print("=" * 70)
    print("PHASE 2 — FINAL TEST METRIC SUMMARY")
    print("=" * 70)

    data = load_test_metrics()

    results = extract_model_metrics(data)
    results = deduplicate_results(results)
    results = normalize_model_names(results)
    results = select_final_models(results)

    if not results:
        raise RuntimeError(
            "Could not identify the selected email and URL models "
            "inside test_metrics.json."
        )

    df = create_summary_table(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV.
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Save JSON.
    json_records = []

    for result in results:
        json_records.append(
            {
                "modality": result["modality"],
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
                "confusion_matrix": result.get(
                    "confusion_matrix"
                ),
            }
        )

    final_json = {
        "experiment": "phase2_final_test_metrics",
        "test_set_used": True,
        "models": json_records,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_json,
            file,
            indent=2,
        )

    print()
    print("FINAL TEST RESULTS")
    print("-" * 70)

    for _, row in df.iterrows():
        print(f"\n{row['Modality']} — {row['Model']}")
        print(f"  Accuracy : {row['Accuracy']:.4f}")
        print(f"  Precision: {row['Precision']:.4f}")
        print(f"  Recall   : {row['Recall']:.4f}")
        print(f"  F1       : {row['F1']:.4f}")
        print(f"  ROC-AUC  : {row['ROC-AUC']:.4f}")
        print(f"  PR-AUC   : {row['PR-AUC']:.4f}")

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV : {OUTPUT_CSV}")

    print()
    print("STEP 17 COMPLETE")


if __name__ == "__main__":
    main()