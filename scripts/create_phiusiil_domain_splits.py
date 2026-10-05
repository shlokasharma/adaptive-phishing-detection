from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/interim/url/phiusiil_cleaned.csv"
)

OUTPUT_DIR = Path(
    "data/splits/url/phiusiil"
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
    print("PHIUSIIL — DOMAIN-DISJOINT DATASET SPLITTING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "sample_id",
        "source_dataset",
        "original_url",
        "clean_url",
        "domain",
        "label",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("[PASS] Required columns found.")

    # --------------------------------------------------------
    # Remove rows with missing domain
    # --------------------------------------------------------

    missing_domains = int(
        df["domain"].isna().sum()
    )

    print(
        f"\nRows with missing domain: "
        f"{missing_domains:,}"
    )

    if missing_domains > 0:

        df = df.dropna(
            subset=["domain"]
        ).copy()

    # --------------------------------------------------------
    # Remove exact duplicate URLs
    # --------------------------------------------------------

    duplicate_urls = int(
        df.duplicated(
            subset=["clean_url"]
        ).sum()
    )

    print(
        f"Duplicate clean URLs: "
        f"{duplicate_urls:,}"
    )

    if duplicate_urls > 0:

        df = df.drop_duplicates(
            subset=["clean_url"],
            keep="first"
        ).reset_index(drop=True)

    print(
        f"Rows after URL deduplication: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Build domain-level table
    # --------------------------------------------------------
    #
    # Every domain will belong to exactly one split.
    #
    # For domains containing both labels, use the majority
    # label for stratification only.
    #
    # The original row-level labels remain unchanged.
    # --------------------------------------------------------

    domain_stats = (
        df.groupby("domain")
        .agg(
            row_count=("sample_id", "count"),
            label_0_count=("label", lambda x: (x == 0).sum()),
            label_1_count=("label", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )

    domain_stats["domain_label"] = (
        domain_stats["label_1_count"]
        >=
        domain_stats["label_0_count"]
    ).astype(int)

    print(
        f"\nUnique domains: "
        f"{len(domain_stats):,}"
    )

    mixed_domains = int(
        (
            (domain_stats["label_0_count"] > 0)
            &
            (domain_stats["label_1_count"] > 0)
        ).sum()
    )

    print(
        f"Mixed-label domains: "
        f"{mixed_domains:,}"
    )

    # --------------------------------------------------------
    # First split: domains → train / temporary
    # --------------------------------------------------------

    train_domains, temp_domains = train_test_split(
        domain_stats,
        test_size=(
            VALIDATION_RATIO + TEST_RATIO
        ),
        stratify=domain_stats["domain_label"],
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Second split: temporary → validation / test
    # --------------------------------------------------------

    validation_domains, test_domains = train_test_split(
        temp_domains,
        test_size=(
            TEST_RATIO
            /
            (VALIDATION_RATIO + TEST_RATIO)
        ),
        stratify=temp_domains["domain_label"],
        random_state=RANDOM_STATE,
    )

    train_domain_set = set(
        train_domains["domain"]
    )

    validation_domain_set = set(
        validation_domains["domain"]
    )

    test_domain_set = set(
        test_domains["domain"]
    )

    # --------------------------------------------------------
    # Assign rows to splits
    # --------------------------------------------------------

    train_df = df[
        df["domain"].isin(train_domain_set)
    ].copy()

    validation_df = df[
        df["domain"].isin(validation_domain_set)
    ].copy()

    test_df = df[
        df["domain"].isin(test_domain_set)
    ].copy()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save datasets
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
    # Report row counts
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
    # Domain counts
    # --------------------------------------------------------

    print("\nDomain counts:")

    print(
        f"TRAIN      : "
        f"{len(train_domain_set):,}"
    )

    print(
        f"VALIDATION : "
        f"{len(validation_domain_set):,}"
    )

    print(
        f"TEST       : "
        f"{len(test_domain_set):,}"
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
    # DOMAIN OVERLAP CHECK
    # --------------------------------------------------------

    train_validation_overlap = (
        train_domain_set
        &
        validation_domain_set
    )

    train_test_overlap = (
        train_domain_set
        &
        test_domain_set
    )

    validation_test_overlap = (
        validation_domain_set
        &
        test_domain_set
    )

    print("\nDomain overlap checks:")

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
    # SAMPLE ID OVERLAP CHECK
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

    id_overlap = (
        len(train_ids & validation_ids)
        +
        len(train_ids & test_ids)
        +
        len(validation_ids & test_ids)
    )

    print("\nSample ID overlap count:")
    print(id_overlap)

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
        and id_overlap == 0
    ):

        print("\nSTATUS: PASS")

        print(
            "No domains or sample IDs overlap "
            "between the splits."
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
    print("PHIUSIIL DOMAIN SPLITTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()