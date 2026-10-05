from pathlib import Path
import pandas as pd


URL_DIR = Path("data/raw/url")


def inspect_csv(file_path: Path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(file_path)

        print(f"Rows       : {len(df):,}")
        print(f"Columns    : {len(df.columns):,}")
        print(f"File size  : {file_path.stat().st_size / (1024 * 1024):.2f} MB")

        print("\nColumns:")
        for i, col in enumerate(df.columns, start=1):
            print(f"{i:3}. {col}")

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("No missing values.")
        else:
            print(missing.sort_values(ascending=False))

        print("\nDuplicate rows:")
        print(df.duplicated().sum())

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nLast 5 rows:")
        print(df.tail())

    except Exception as e:
        print(f"ERROR: {e}")


def main():
    print("=" * 80)
    print("URL DATASET INSPECTION")
    print("=" * 80)

    files = sorted(URL_DIR.glob("*.csv"))

    if not files:
        print("No CSV files found.")
        return

    for file_path in files:
        inspect_csv(file_path)

    print("\n" + "=" * 80)
    print("URL DATASET INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()