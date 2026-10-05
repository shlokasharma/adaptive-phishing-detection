from pathlib import Path
import pandas as pd
import re


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_EMAIL_DIR = PROJECT_ROOT / "data" / "raw" / "email"
PROCESSED_EMAIL_DIR = PROJECT_ROOT / "data" / "processed" / "email"

PROCESSED_EMAIL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def clean_text(value):
    """
    Convert a value into clean text.

    Missing values become an empty string.
    Whitespace is normalized.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_column(columns, candidates):
    """
    Find the first matching column from a list of candidates.
    Matching is case-insensitive.
    """

    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        candidate = candidate.lower()

        if candidate in normalized:
            return normalized[candidate]

    return None


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):
    """
    Convert dataset-specific labels into:

        1 = phishing / malicious
        0 = legitimate / benign

    Unknown labels remain as None.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    phishing_labels = {
        "1",
        "1.0",
        "phishing",
        "phish",
        "spam",
        "fraud",
        "malicious",
        "malware",
        "scam",
        "junk",
    }

    legitimate_labels = {
        "0",
        "0.0",
        "ham",
        "legitimate",
        "benign",
        "normal",
        "safe",
        "clean",
    }

    if value in phishing_labels:
        return 1

    if value in legitimate_labels:
        return 0

    return None


# ============================================================
# DATASET NORMALIZATION
# ============================================================

def normalize_dataset(file_path):
    """
    Read one raw email dataset and convert it into
    the canonical schema.
    """

    print("\n" + "=" * 80)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
            encoding_errors="replace"
        )

    except Exception as error:
        print(f"ERROR reading {file_path.name}: {error}")
        return None

    print(f"Original shape: {df.shape}")

    columns = list(df.columns)

    # --------------------------------------------------------
    # Detect likely columns
    # --------------------------------------------------------

    label_column = find_column(
        columns,
        [
            "label",
            "class",
            "target",
            "category",
            "type",
            "spam",
        ],
    )

    subject_column = find_column(
        columns,
        [
            "subject",
            "email_subject",
            "title",
        ],
    )

    body_column = find_column(
        columns,
        [
            "body",
            "email_body",
            "message",
            "text",
            "content",
        ],
    )

    sender_column = find_column(
        columns,
        [
            "sender",
            "from",
            "email_from",
        ],
    )

    receiver_column = find_column(
        columns,
        [
            "receiver",
            "recipient",
            "to",
            "email_to",
        ],
    )

    timestamp_column = find_column(
        columns,
        [
            "date",
            "timestamp",
            "time",
        ],
    )

    print("\nDetected columns:")

    print(f"  Label     : {label_column}")
    print(f"  Subject   : {subject_column}")
    print(f"  Body      : {body_column}")
    print(f"  Sender    : {sender_column}")
    print(f"  Receiver  : {receiver_column}")
    print(f"  Timestamp : {timestamp_column}")

    # --------------------------------------------------------
    # Create canonical dataframe
    # --------------------------------------------------------

    normalized = pd.DataFrame()

    normalized["sample_id"] = [
        f"{file_path.stem}_{i:06d}"
        for i in range(len(df))
    ]

    normalized["source_dataset"] = file_path.stem

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if label_column is not None:
        normalized["label"] = df[label_column].apply(
            normalize_label
        )
    else:
        normalized["label"] = None

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    if subject_column is not None:
        normalized["subject"] = df[subject_column].apply(
            clean_text
        )
    else:
        normalized["subject"] = ""

    # --------------------------------------------------------
    # Body
    # --------------------------------------------------------

    if body_column is not None:
        normalized["body"] = df[body_column].apply(
            clean_text
        )
    else:
        normalized["body"] = ""

    # --------------------------------------------------------
    # Sender
    # --------------------------------------------------------

    if sender_column is not None:
        normalized["sender"] = df[sender_column].apply(
            clean_text
        )
    else:
        normalized["sender"] = ""

    # --------------------------------------------------------
    # Receiver
    # --------------------------------------------------------

    if receiver_column is not None:
        normalized["receiver"] = df[receiver_column].apply(
            clean_text
        )
    else:
        normalized["receiver"] = ""

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if timestamp_column is not None:
        normalized["timestamp"] = df[timestamp_column].apply(
            clean_text
        )
    else:
        normalized["timestamp"] = ""

    # --------------------------------------------------------
    # Combined text
    # --------------------------------------------------------

    normalized["raw_text"] = (
        normalized["subject"]
        + " "
        + normalized["body"]
    ).str.strip()

    # --------------------------------------------------------
    # Remove completely empty records
    # --------------------------------------------------------

    before = len(normalized)

    normalized = normalized[
        normalized["raw_text"].str.len() > 0
    ].copy()

    removed_empty = before - len(normalized)

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    before = len(normalized)

    normalized = normalized.drop_duplicates(
        subset=["raw_text"],
        keep="first"
    ).copy()

    removed_duplicates = before - len(normalized)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nCleaning results:")

    print(f"  Empty records removed     : {removed_empty}")
    print(f"  Exact duplicates removed : {removed_duplicates}")
    print(f"  Final records            : {len(normalized)}")

    print("\nLabel distribution:")

    print(
        normalized["label"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        PROCESSED_EMAIL_DIR
        / f"{file_path.stem}_normalized.csv"
    )

    normalized.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved:")
    print(output_file)

    return normalized


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EMAIL DATASET NORMALIZATION")
    print("=" * 80)

    csv_files = sorted(
        RAW_EMAIL_DIR.glob("*.csv")
    )

    if not csv_files:
        print("\nNo email CSV files found.")
        return

    successful = 0

    for file_path in csv_files:

        result = normalize_dataset(file_path)

        if result is not None:
            successful += 1

    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)

    print(f"\nDatasets processed successfully: {successful}")
    print(
        f"Output directory:\n"
        f"{PROCESSED_EMAIL_DIR}"
    )


if __name__ == "__main__":
    main()