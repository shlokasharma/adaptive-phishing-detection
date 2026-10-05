from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/url/uci_phishing_websites_features.csv"
)

OUTPUT_DIR = Path("data/splits/url/uci_features")

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("UCI PHISHING WEBSITES — LEAKAGE-SAFE SPLITTING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput rows: {len(df):,}")

    # --------------------------------------------------------
    # Remove duplicate feature vectors
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in df.columns
        if col not in ["sample_id", "source_dataset", "label"]
    ]

    duplicate_count = df.duplicated(
        subset=feature_columns
    ).sum()

    print(
        f"Duplicate feature vectors found: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        df = df.drop_duplicates(
            subset=feature_columns,
            keep="first"
        ).reset_index(drop=True)

    print(
        f"Rows after duplicate removal: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(
            TEST_SIZE /
            (VALIDATION_SIZE + TEST_SIZE)
        ),
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SPLIT RESULTS")
    print("=" * 80)

    print(
        f"\nTrain      : {len(train_df):,} "
        f"({len(train_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Validation : {len(validation_df):,} "
        f"({len(validation_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Test       : {len(test_df):,} "
        f"({len(test_df) / len(df) * 100:.2f}%)"
    )

    print("\nLabel distribution:")

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:

        counts = (
            split_df["label"]
            .value_counts()
            .sort_index()
        )

        print(f"\n{name}")

        for label, count in counts.items():

            label_name = (
                "Legitimate"
                if label == 0
                else "Phishing"
            )

            print(
                f"  {label} ({label_name}): "
                f"{count:,}"
            )

    # --------------------------------------------------------
    # Final overlap check
    # --------------------------------------------------------

    train_ids = set(train_df["sample_id"])
    validation_ids = set(validation_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    train_validation_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    validation_test_overlap = (
        validation_ids & test_ids
    )

    print("\nSplit overlap checks:")

    print(
        f"Train ∩ Validation: "
        f"{len(train_validation_overlap)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(validation_test_overlap)}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if not (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):

        print("\nSTATUS: PASS")
        print(
            "No sample IDs overlap between splits."
        )

    else:

        print("\nSTATUS: FAIL")
        print(
            "Sample overlap detected."
        )

    print("\nSaved files:")

    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")

    print("\n" + "=" * 80)
    print("UCI FEATURE SPLITTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()