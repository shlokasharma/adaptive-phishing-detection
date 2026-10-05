from pathlib import Path
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMAIL_DIR = PROJECT_ROOT / "data" / "raw" / "email"
URL_DIR = PROJECT_ROOT / "data" / "raw" / "url"

OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "dataset_inspection_summary.csv"


# ============================================================
# Dataset inspection
# ============================================================

def inspect_csv(file_path: Path) -> dict:
    """Inspect a CSV dataset and return summary statistics."""

    print("\n" + "=" * 70)
    print(f"DATASET: {file_path.name}")
    print("=" * 70)

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
            encoding_errors="replace",
        )

    except Exception as error:
        print(f"[ERROR] Could not read {file_path.name}")
        print(error)

        return {
            "dataset": file_path.name,
            "path": str(file_path),
            "status": "ERROR",
            "rows": None,
            "columns": None,
            "missing_values": None,
            "duplicate_rows": None,
        }

    rows, columns = df.shape

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    print(f"Rows:              {rows:,}")
    print(f"Columns:           {columns:,}")
    print(f"Missing values:    {missing_values:,}")
    print(f"Duplicate rows:    {duplicate_rows:,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    return {
        "dataset": file_path.name,
        "path": str(file_path),
        "status": "OK",
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE PHISHING DETECTION")
    print("DATASET INSPECTION")
    print("=" * 70)

    datasets = []

    # --------------------------------------------------------
    # Email CSV files
    # --------------------------------------------------------

    email_files = sorted(EMAIL_DIR.glob("*.csv"))

    print(f"\nEmail CSV files found: {len(email_files)}")

    for file_path in email_files:
        result = inspect_csv(file_path)
        result["modality"] = "email"
        datasets.append(result)

    # --------------------------------------------------------
    # URL CSV files
    # --------------------------------------------------------

    url_files = sorted(URL_DIR.glob("*.csv"))

    print(f"\nURL CSV files found: {len(url_files)}")

    for file_path in url_files:
        result = inspect_csv(file_path)
        result["modality"] = "url"
        datasets.append(result)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    if datasets:

        summary_df = pd.DataFrame(datasets)

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
        )

        print("\n" + "=" * 70)
        print("INSPECTION COMPLETE")
        print("=" * 70)

        print(f"\nSummary saved to:")
        print(SUMMARY_FILE)

        print("\nDataset summary:")
        print(
            summary_df[
                [
                    "dataset",
                    "modality",
                    "status",
                    "rows",
                    "columns",
                    "missing_values",
                    "duplicate_rows",
                ]
            ].to_string(index=False)
        )

    else:
        print("\n[WARNING] No CSV datasets were found.")


if __name__ == "__main__":
    main()