from pathlib import Path
import pandas as pd


URL_DIR = Path("data/processed/url")


EXPECTED_COLUMNS = [
    "sample_id",
    "source_dataset",
    "url",
    "label",
]


def validate_dataset(file_path: Path):

    print("\n" + "=" * 80)
    print(f"VALIDATING: {file_path.name}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    errors = []
    warnings = []

    # ---------------------------------------------------------
    # 1. Shape
    # ---------------------------------------------------------

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # ---------------------------------------------------------
    # 2. Column validation
    # ---------------------------------------------------------

    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(
            f"Unexpected columns: {list(df.columns)}"
        )

    print("\nColumns:")
    print(list(df.columns))

    # ---------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")

    missing = df.isnull().sum()

    print(missing)

    for column, count in missing.items():
        if count > 0:
            errors.append(
                f"{column} contains {count:,} missing values"
            )

    # ---------------------------------------------------------
    # 4. Empty URLs
    # ---------------------------------------------------------

    empty_urls = (
        df["url"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"\nEmpty URLs: {empty_urls:,}")

    if empty_urls > 0:
        errors.append(
            f"{empty_urls:,} empty URLs found"
        )

    # ---------------------------------------------------------
    # 5. Label validation
    # ---------------------------------------------------------

    unique_labels = sorted(df["label"].dropna().unique())

    print("\nUnique labels:")
    print(unique_labels)

    invalid_labels = set(unique_labels) - {0, 1}

    if invalid_labels:
        errors.append(
            f"Invalid labels found: {invalid_labels}"
        )

    # ---------------------------------------------------------
    # 6. Label distribution
    # ---------------------------------------------------------

    print("\nLabel distribution:")

    label_counts = df["label"].value_counts().sort_index()

    print(label_counts)

    # ---------------------------------------------------------
    # 7. Duplicate rows
    # ---------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    print(f"\nDuplicate complete rows: {duplicate_rows:,}")

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows:,} duplicate complete rows found"
        )

    # ---------------------------------------------------------
    # 8. Duplicate URLs
    # ---------------------------------------------------------

    duplicate_urls = df["url"].duplicated().sum()

    print(f"Duplicate URLs: {duplicate_urls:,}")

    if duplicate_urls > 0:
        warnings.append(
            f"{duplicate_urls:,} duplicate URLs found"
        )

    # ---------------------------------------------------------
    # 9. Conflicting labels
    # ---------------------------------------------------------

    url_label_counts = (
        df.groupby("url")["label"]
        .nunique()
    )

    conflicting_urls = (
        url_label_counts > 1
    ).sum()

    print(
        f"URLs appearing with conflicting labels: "
        f"{conflicting_urls:,}"
    )

    if conflicting_urls > 0:
        warnings.append(
            f"{conflicting_urls:,} URLs have conflicting labels"
        )

    # ---------------------------------------------------------
    # 10. Very short URLs
    # ---------------------------------------------------------

    short_urls = (
        df["url"]
        .astype(str)
        .str.len()
        .lt(10)
        .sum()
    )

    print(f"URLs shorter than 10 characters: {short_urls:,}")

    if short_urls > 0:
        warnings.append(
            f"{short_urls:,} unusually short URLs found"
        )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n" + "-" * 80)

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  [ERROR] {error}")
    else:
        print("ERRORS: None")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  [WARNING] {warning}")
    else:
        print("\nWARNINGS: None")

    print("-" * 80)

    if errors:
        print("STATUS: FAILED")
        return False

    print("STATUS: PASSED")
    return True


def main():

    print("=" * 80)
    print("URL DATASET VALIDATION")
    print("=" * 80)

    files = sorted(
        URL_DIR.glob("*_normalized.csv")
    )

    if not files:
        print("No normalized URL datasets found.")
        return

    results = []

    for file_path in files:
        result = validate_dataset(file_path)
        results.append(result)

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    for file_path, result in zip(files, results):

        status = "PASSED" if result else "FAILED"

        print(
            f"{file_path.name:<50} {status}"
        )

    if all(results):
        print("\nALL URL DATASETS PASSED VALIDATION.")
    else:
        print("\nSOME URL DATASETS FAILED VALIDATION.")

    print("=" * 80)


if __name__ == "__main__":
    main()