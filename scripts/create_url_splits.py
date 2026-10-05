from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_DIR = Path("data/interim/url")
OUTPUT_DIR = Path("data/splits/url")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def create_split(df, dataset_name):
    """
    Create stratified train/validation/test splits.

    Split ratio:
        70% train
        15% validation
        15% test
    """

    print("\n" + "=" * 80)
    print(f"CREATING SPLITS: {dataset_name}")
    print("=" * 80)

    # ---------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # ---------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # ---------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    # ---------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    train_file = OUTPUT_DIR / f"{dataset_name}_train.csv"
    val_file = OUTPUT_DIR / f"{dataset_name}_validation.csv"
    test_file = OUTPUT_DIR / f"{dataset_name}_test.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print(f"\nTotal      : {len(df):,}")
    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(val_df):,}")
    print(f"Test       : {len(test_df):,}")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts(normalize=True).sort_index())

    print("\nValidation label distribution:")
    print(val_df["label"].value_counts(normalize=True).sort_index())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts(normalize=True).sort_index())

    return train_df, val_df, test_df


def check_split_overlap(train_df, val_df, test_df, dataset_name):
    """
    Verify that exact clean URLs do not appear across splits.
    """

    print("\n" + "-" * 80)
    print(f"CHECKING SPLIT LEAKAGE: {dataset_name}")
    print("-" * 80)

    train_urls = set(train_df["clean_url"])
    val_urls = set(val_df["clean_url"])
    test_urls = set(test_df["clean_url"])

    train_val = train_urls.intersection(val_urls)
    train_test = train_urls.intersection(test_urls)
    val_test = val_urls.intersection(test_urls)

    print(f"Train ∩ Validation : {len(train_val):,}")
    print(f"Train ∩ Test       : {len(train_test):,}")
    print(f"Validation ∩ Test  : {len(val_test):,}")

    if train_val or train_test or val_test:
        print("WARNING: URL overlap detected!")
    else:
        print("No exact URL overlap detected.")


def process_dataset(file_path):

    dataset_name = file_path.stem.replace(
        "_cleaned",
        ""
    )

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # Handle conflicting labels
    # ---------------------------------------------------------

    label_counts = (
        df.groupby("clean_url")["label"]
        .nunique()
    )

    conflicting_urls = label_counts[
        label_counts > 1
    ].index

    if len(conflicting_urls) > 0:

        print(
            f"\nFound {len(conflicting_urls)} "
            "conflicting URLs."
        )

        print(
            "These records will be excluded from "
            "the model split."
        )

        df = df[
            ~df["clean_url"].isin(conflicting_urls)
        ].copy()

    # ---------------------------------------------------------
    # Create splits
    # ---------------------------------------------------------

    train_df, val_df, test_df = create_split(
        df,
        dataset_name
    )

    # ---------------------------------------------------------
    # Check overlap
    # ---------------------------------------------------------

    check_split_overlap(
        train_df,
        val_df,
        test_df,
        dataset_name
    )


def main():

    print("=" * 80)
    print("URL TRAIN / VALIDATION / TEST SPLITTING")
    print("=" * 80)

    files = sorted(
        INPUT_DIR.glob("*_cleaned.csv")
    )

    if not files:
        print("No cleaned URL datasets found.")
        return

    for file_path in files:
        process_dataset(file_path)

    print("\n" + "=" * 80)
    print("URL SPLITTING COMPLETE")
    print("=" * 80)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()