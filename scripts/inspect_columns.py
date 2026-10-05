from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMAIL_DIR = PROJECT_ROOT / "data" / "raw" / "email"


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

        print(f"\nShape: {df.shape}")

        print("\nColumns:")
        for i, column in enumerate(df.columns, start=1):
            print(f"  {i}. {column}")

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"\nERROR reading {file_path.name}")
        print(f"Reason: {e}")


def main():
    print("=" * 80)
    print("EMAIL DATASET COLUMN INSPECTION")
    print("=" * 80)

    csv_files = sorted(EMAIL_DIR.glob("*.csv"))

    if not csv_files:
        print("\nNo CSV files found.")
        return

    for file_path in csv_files:
        inspect_dataset(file_path)

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()