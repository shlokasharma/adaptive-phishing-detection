from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_EMAIL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "email"
)


def inspect_dataset(file_path):

    print("\n" + "=" * 80)
    print(f"DATASET: {file_path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
            encoding_errors="replace"
        )

    except Exception as error:
        print(f"ERROR: {error}")
        return

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nLabel distribution:")

    if "label" in df.columns:
        print(
            df["label"]
            .value_counts(dropna=False)
            .sort_index()
        )

    print("\nMissing values:")

    print(
        df.isnull()
        .sum()
        .to_string()
    )

    print("\nEmpty values:")

    for column in df.columns:
        if df[column].dtype == "object":
            empty_count = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            print(
                f"  {column}: {empty_count:,}"
            )

    print("\nDuplicate rows:")

    print(
        f"  Exact duplicates: "
        f"{df.duplicated().sum():,}"
    )

    if "raw_text" in df.columns:

        print("\nText statistics:")

        text_lengths = (
            df["raw_text"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        print(
            f"  Minimum length : {text_lengths.min():,}"
        )

        print(
            f"  Maximum length : {text_lengths.max():,}"
        )

        print(
            f"  Mean length    : {text_lengths.mean():.2f}"
        )

        print(
            f"  Empty text     : "
            f"{(text_lengths == 0).sum():,}"
        )

    print("\nFirst 2 records:")

    print(
        df.head(2).to_string(index=False)
    )


def main():

    print("=" * 80)
    print("PROCESSED EMAIL DATASET VALIDATION")
    print("=" * 80)

    files = sorted(
        PROCESSED_EMAIL_DIR.glob(
            "*_normalized.csv"
        )
    )

    if not files:
        print("\nNo normalized datasets found.")
        return

    print(
        f"\nFound {len(files)} normalized datasets."
    )

    for file_path in files:
        inspect_dataset(file_path)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()