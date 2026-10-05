from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "data/interim/email"
)

OUTPUT_DIR = Path(
    "data/splits/email"
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EMAIL DATASET — LEAKAGE-SAFE SPLITTING")
    print("=" * 80)

    files = sorted(
        INPUT_DIR.glob("*_cleaned.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No cleaned email datasets found in {INPUT_DIR}"
        )

    all_data = []

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    for file in files:

        df = pd.read_csv(file)

        required_columns = [
            "sample_id",
            "source_dataset",
            "label",
            "clean_text",
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:
            print(
                f"[WARNING] Skipping {file.name}. "
                f"Missing columns: {missing}"
            )
            continue

        df = df[
            required_columns
        ].copy()

        all_data.append(df)

    if not all_data:
        raise ValueError(
            "No valid email datasets found."
        )

    df = pd.concat(
        all_data,
        ignore_index=True
    )

    print(
        f"\nTotal rows loaded: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Remove missing text
    # --------------------------------------------------------

    missing_text = int(
        df["clean_text"].isna().sum()
    )

    print(
        f"Rows with missing clean_text: "
        f"{missing_text:,}"
    )

    if missing_text > 0:

        df = df.dropna(
            subset=["clean_text"]
        ).copy()

    # --------------------------------------------------------
    # Create text groups
    # --------------------------------------------------------
    #
    # Every identical clean_text gets the same group ID.
    #
    # This prevents identical emails from appearing in
    # different splits.
    # --------------------------------------------------------

    df["text_group"] = (
        pd.factorize(
            df["clean_text"]
        )[0]
    )

    group_table = (
        df.groupby("text_group")
        .agg(
            label=("label", "first"),
            row_count=("sample_id", "count"),
        )
        .reset_index()
    )

    print(
        f"\nUnique text groups: "
        f"{len(group_table):,}"
    )

    # --------------------------------------------------------
    # Group-level stratification
    # --------------------------------------------------------

    train_groups, temp_groups = train_test_split(
        group_table,
        test_size=(
            VALIDATION_RATIO + TEST_RATIO
        ),
        stratify=group_table["label"],
        random_state=RANDOM_STATE,
    )

    validation_groups, test_groups = train_test_split(
        temp_groups,
        test_size=(
            TEST_RATIO
            /
            (VALIDATION_RATIO + TEST_RATIO)
        ),
        stratify=temp_groups["label"],
        random_state=RANDOM_STATE,
    )

    train_group_ids = set(
        train_groups["text_group"]
    )

    validation_group_ids = set(
        validation_groups["text_group"]
    )

    test_group_ids = set(
        test_groups["text_group"]
    )

    # --------------------------------------------------------
    # Assign rows
    # --------------------------------------------------------

    train_df = df[
        df["text_group"].isin(
            train_group_ids
        )
    ].copy()

    validation_df = df[
        df["text_group"].isin(
            validation_group_ids
        )
    ].copy()

    test_df = df[
        df["text_group"].isin(
            test_group_ids
        )
    ].copy()

    # --------------------------------------------------------
    # Remove helper column from final files
    # --------------------------------------------------------

    train_df = train_df.drop(
        columns=["text_group"]
    )

    validation_df = validation_df.drop(
        columns=["text_group"]
    )

    test_df = test_df.drop(
        columns=["text_group"]
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
    print("ROW-LEVEL SPLIT RESULTS")
    print("=" * 80)

    total_rows = len(df)

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:

        percentage = (
            len(split_df)
            /
            total_rows
            *
            100
        )

        print(
            f"{name:12s}: "
            f"{len(split_df):,} rows "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    print("\nLabel distributions:")

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:

        print(f"\n{name}")

        counts = (
            split_df["label"]
            .value_counts()
            .sort_index()
        )

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
    # Text overlap check
    # --------------------------------------------------------

    train_texts = set(
        train_df["clean_text"]
    )

    validation_texts = set(
        validation_df["clean_text"]
    )

    test_texts = set(
        test_df["clean_text"]
    )

    train_validation_overlap = (
        train_texts
        &
        validation_texts
    )

    train_test_overlap = (
        train_texts
        &
        test_texts
    )

    validation_test_overlap = (
        validation_texts
        &
        test_texts
    )

    print("\nText overlap checks:")

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
    # Sample ID overlap check
    # --------------------------------------------------------

    train_ids = set(
        train_df["sample_id"]
    )

    validation_ids = set(
        validation_df["sample_id"]
    )

    test_ids = set(
        test_df["sample_id"]
    )

    sample_overlap = (
        len(train_ids & validation_ids)
        +
        len(train_ids & test_ids)
        +
        len(validation_ids & test_ids)
    )

    print(
        "\nSample ID overlap count:"
    )

    print(sample_overlap)

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
        and sample_overlap == 0
    ):

        print("\nSTATUS: PASS")

        print(
            "No identical email texts or sample IDs "
            "overlap between splits."
        )

    else:

        print("\nSTATUS: FAIL")

        print(
            "Leakage detected between splits."
        )

    print("\nSaved files:")

    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")

    print("\n" + "=" * 80)
    print("EMAIL SPLITTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()