"""
Select the strongest baseline model for each modality.

Model selection is based exclusively on validation F1-score.
The held-out test set is not used for model selection.

Input:
    experiments/baselines/outputs/hyperparameter_tuning_results.json

Output:
    experiments/baselines/outputs/selected_baseline_models.json
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
    / "hyperparameter_tuning_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
    / "selected_baseline_models.json"
)


def load_tuning_results() -> dict:
    """Load the completed hyperparameter tuning results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Tuning results not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_best_models(results: dict) -> dict:
    """
    Select the strongest model for each modality.

    Selection criterion:
        Highest validation F1-score.

    The test set is never accessed.
    """

    selected_models = {}

    for modality in ("email", "url"):

        if modality not in results:
            raise ValueError(
                f"Missing modality in tuning results: {modality}"
            )

        modality_results = results[modality]

        candidates = []

        for model_name, model_data in modality_results.items():

            if not isinstance(model_data, dict):
                continue

            if "best_result" not in model_data:
                continue

            best_result = model_data["best_result"]

            if "f1" not in best_result:
                continue

            candidates.append(
                {
                    "model_name": model_data.get(
                        "model_name",
                        model_name,
                    ),
                    "validation_f1": float(
                        best_result["f1"]
                    ),
                    "parameters": best_result.get(
                        "parameters",
                        model_data.get(
                            "best_result",
                            {},
                        ).get(
                            "parameters",
                            {},
                        ),
                    ),
                    "accuracy": float(
                        best_result.get(
                            "accuracy",
                            0.0,
                        )
                    ),
                    "precision": float(
                        best_result.get(
                            "precision",
                            0.0,
                        )
                    ),
                    "recall": float(
                        best_result.get(
                            "recall",
                            0.0,
                        )
                    ),
                    "roc_auc": float(
                        best_result.get(
                            "roc_auc",
                            0.0,
                        )
                    ),
                    "pr_auc": float(
                        best_result.get(
                            "pr_auc",
                            0.0,
                        )
                    ),
                    "training_time_seconds": float(
                        best_result.get(
                            "training_time_seconds",
                            0.0,
                        )
                    ),
                    "inference_time_seconds": float(
                        best_result.get(
                            "inference_time_seconds",
                            0.0,
                        )
                    ),
                    "search_status": model_data.get(
                        "search_status",
                        "unknown",
                    ),
                }
            )

        if not candidates:
            raise ValueError(
                f"No valid model candidates found for modality: "
                f"{modality}"
            )

        best_model = max(
            candidates,
            key=lambda model: model["validation_f1"],
        )

        selected_models[modality] = best_model

    return selected_models


def main() -> None:
    """Select and save the strongest baseline models."""

    results = load_tuning_results()

    selected_models = select_best_models(results)

    output = {
        "experiment": "phase2_baseline_model_selection",
        "selection_criterion": "highest_validation_f1",
        "test_set_used_for_selection": False,
        "selected_models": selected_models,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("=" * 70)
    print("SELECTED BASELINE MODELS")
    print("=" * 70)

    for modality, model in selected_models.items():

        print(
            f"{modality.upper()}: "
            f"{model['model_name']}"
        )

        print(
            f"  Validation F1: "
            f"{model['validation_f1']:.4f}"
        )

        print(
            f"  Accuracy: "
            f"{model['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{model['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{model['recall']:.4f}"
        )

        print(
            f"  ROC-AUC: "
            f"{model['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC: "
            f"{model['pr_auc']:.4f}"
        )

        print(
            f"  Parameters: "
            f"{model['parameters']}"
        )

        print()

    print("=" * 70)
    print(
        f"Selection results saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()