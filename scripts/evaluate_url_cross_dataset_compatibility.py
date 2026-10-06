"""
Assess cross-dataset feature compatibility for URL models.

The Phase 2 selected URL model is trained on the UCI Phishing
Websites feature representation.

This script compares the frozen UCI feature schema with the
PhiUSIIL schema before any cross-dataset evaluation is attempted.

No model is retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UCI_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "train.csv"
)

PHIUSIIL_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "phiusiil_train.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
    / "url_cross_dataset_compatibility.json"
)


EXCLUDED_COLUMNS = {
    "sample_id",
    "source_dataset",
    "label",
}


def get_feature_columns(path: Path) -> list[str]:
    """Return numerical model feature columns."""

    df = pd.read_csv(path)

    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def main() -> None:
    """Compare UCI and PhiUSIIL feature schemas."""

    print("=" * 70)
    print("PHASE 2 — URL CROSS-DATASET COMPATIBILITY")
    print("=" * 70)

    if not UCI_TRAIN.exists():
        raise FileNotFoundError(
            f"UCI training file not found:\n{UCI_TRAIN}"
        )

    if not PHIUSIIL_TRAIN.exists():
        raise FileNotFoundError(
            f"PhiUSIIL training file not found:\n{PHIUSIIL_TRAIN}"
        )

    uci_features = get_feature_columns(UCI_TRAIN)
    phiusiil_features = get_feature_columns(PHIUSIIL_TRAIN)

    common_features = sorted(
        set(uci_features) & set(phiusiil_features)
    )

    only_uci = sorted(
        set(uci_features) - set(phiusiil_features)
    )

    only_phiusiil = sorted(
        set(phiusiil_features) - set(uci_features)
    )

    compatible = (
        len(uci_features) == len(phiusiil_features)
        and uci_features == phiusiil_features
    )

    result = {
        "experiment": (
            "phase2_url_cross_dataset_feature_compatibility"
        ),
        "source_model_training_dataset": "UCI Phishing Websites",
        "external_evaluation_dataset": "PhiUSIIL",
        "uci_feature_count": len(uci_features),
        "phiusiil_feature_count": len(phiusiil_features),
        "common_feature_count": len(common_features),
        "uci_only_feature_count": len(only_uci),
        "phiusiil_only_feature_count": len(only_phiusiil),
        "schemas_directly_compatible": compatible,
        "direct_cross_dataset_evaluation_valid": compatible,
        "note": (
            "The selected UCI URL Random Forest cannot be directly "
            "evaluated on PhiUSIIL unless the external dataset is "
            "transformed into the identical UCI feature representation. "
            "No such transformation is assumed in Phase 2."
        ),
        "uci_features": uci_features,
        "phiusiil_features": phiusiil_features,
        "common_features": common_features,
        "uci_only_features": only_uci,
        "phiusiil_only_features": only_phiusiil,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"\nUCI feature count      : {len(uci_features)}"
    )
    print(
        f"PhiUSIIL feature count : {len(phiusiil_features)}"
    )
    print(
        f"Common features        : {len(common_features)}"
    )

    print(
        "\nDirectly compatible:",
        compatible,
    )

    print(
        "\nOutput:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()