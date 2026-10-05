from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "data/interim/email"
)

OUTPUT_DIR = Path(
    "data/metadata"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EMAIL DATASET LEAKAGE AUDIT")
    print("=" * 80)

    files = sorted(
        INPUT_DIR.glob("*_cleaned.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No cleaned email datasets found in {INPUT_DIR}"
        )

    print(
        f"\nCleaned email datasets found: "
        f"{len(files)}"
    )

    all_data = []

    # --------------------------------------------------------
    # Inspect each dataset
    # --------------------------------------------------------

    for file in files:

        print(f"\nProcessing: {file.name}")

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
                f"[WARNING] Missing columns: {missing}"
            )

            continue

        df = df[
            required_columns
        ].copy()

        df["source_file"] = file.name

        all_data.append(df)

        print(
            f"  Rows: {len(df):,}"
        )

        print(
            f"  Labels:\n"
            f"{df['label'].value_counts().sort_index()}"
        )

    if not all_data:
        raise ValueError(
            "No valid cleaned email datasets were found."
        )

    # --------------------------------------------------------
    # Combine ONLY for leakage auditing
    # --------------------------------------------------------

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    print("\n" + "=" * 80)
    print("COMBINED AUDIT DATASET")
    print("=" * 80)

    print(
        f"\nTotal rows: "
        f"{len(combined):,}"
    )

    print(
        f"Unique sample IDs: "
        f"{combined['sample_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    print(
        combined[
            [
                "sample_id",
                "source_dataset",
                "label",
                "clean_text",
            ]
        ]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # Duplicate clean text
    # --------------------------------------------------------

    duplicate_text_count = int(
        combined.duplicated(
            subset=["clean_text"]
        ).sum()
    )

    unique_text_count = (
        combined["clean_text"]
        .nunique()
    )

    print("\nText duplication:")
    print(
        f"  Unique clean texts: "
        f"{unique_text_count:,}"
    )

    print(
        f"  Duplicate clean texts: "
        f"{duplicate_text_count:,}"
    )

    # --------------------------------------------------------
    # Cross-dataset duplicates
    # --------------------------------------------------------

    text_source_counts = (
        combined.groupby("clean_text")[
            "source_dataset"
        ]
        .nunique()
    )

    cross_dataset_text = text_source_counts[
        text_source_counts > 1
    ]

    print(
        "\nCross-dataset duplicate texts: "
        f"{len(cross_dataset_text):,}"
    )

    if len(cross_dataset_text) > 0:

        cross_dataset_df = (
            combined[
                combined["clean_text"].isin(
                    cross_dataset_text.index
                )
            ]
            .sort_values("clean_text")
        )

        output_file = (
            OUTPUT_DIR /
            "email_cross_dataset_duplicates.csv"
        )

        cross_dataset_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"Cross-dataset duplicate report saved to:"
            f"\n  {output_file}"
        )

    # --------------------------------------------------------
    # Conflicting labels
    # --------------------------------------------------------

    text_label_counts = (
        combined.groupby("clean_text")[
            "label"
        ]
        .nunique()
    )

    conflicting_text = text_label_counts[
        text_label_counts > 1
    ]

    print(
        "\nConflicting-label email texts: "
        f"{len(conflicting_text):,}"
    )

    if len(conflicting_text) > 0:

        conflict_df = (
            combined[
                combined["clean_text"].isin(
                    conflicting_text.index
                )
            ]
            .sort_values("clean_text")
        )

        output_file = (
            OUTPUT_DIR /
            "email_conflicting_labels.csv"
        )

        conflict_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"Conflict report saved to:"
            f"\n  {output_file}"
        )

    # --------------------------------------------------------
    # Per-dataset summary
    # --------------------------------------------------------

    dataset_summary = (
        combined
        .groupby("source_dataset")
        .agg(
            rows=("sample_id", "count"),
            unique_texts=("clean_text", "nunique"),
            duplicate_texts=(
                "clean_text",
                lambda x: x.duplicated().sum()
            ),
        )
        .reset_index()
    )

    summary_file = (
        OUTPUT_DIR /
        "email_leakage_summary.csv"
    )

    dataset_summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"\nDataset summary saved to:"
        f"\n  {summary_file}"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print("\nOverall label distribution:")

    label_counts = (
        combined["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in label_counts.items():

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
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EMAIL LEAKAGE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()