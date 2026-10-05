from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw/url")
PROCESSED_DIR = Path("data/processed/url")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_name(column):
    """Convert column names into a consistent format."""
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_url_column(df):
    """Try to identify the URL column."""
    candidates = [
        "url",
        "urls",
        "website",
        "webpage",
        "domain",
        "link",
    ]

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # Fallback: search for columns containing 'url'
    for col in df.columns:
        if "url" in normalize_column_name(col):
            return col

    return None


def find_label_column(df):
    """Try to identify the target/label column."""
    candidates = [
        "label",
        "labels",
        "class",
        "target",
        "result",
        "status",
        "phishing",
    ]

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def normalize_uci_phishing():
    """Normalize UCI Phishing Websites dataset."""

    input_file = RAW_DIR / "uci_phishing_websites.csv"

    if not input_file.exists():
        print(f"Missing file: {input_file}")
        return

    print("\n" + "=" * 80)
    print("PROCESSING UCI PHISHING WEBSITES")
    print("=" * 80)

    df = pd.read_csv(input_file)

    url_column = find_url_column(df)
    label_column = find_label_column(df)

    print(f"Detected URL column   : {url_column}")
    print(f"Detected label column : {label_column}")

    if url_column is None or label_column is None:
        print("\nERROR: Could not automatically identify URL/label columns.")
        print("Available columns:")
        print(list(df.columns))
        return

    output = pd.DataFrame()

    output["url"] = df[url_column].astype(str).str.strip()

    # UCI Phishing Websites uses:
    # -1 = legitimate
    #  1 = phishing
    #
    # Convert to project convention:
    # 0 = legitimate
    # 1 = phishing

    def convert_label(value):
        value_str = str(value).strip().lower()

        if value_str in {"1", "1.0", "phishing", "phish", "malicious"}:
            return 1

        if value_str in {"-1", "-1.0", "0", "0.0", "legitimate", "benign"}:
            return 0

        return None

    output["label"] = df[label_column].apply(convert_label)

    output["source_dataset"] = "uci_phishing_websites"

    output["sample_id"] = [
        f"uci_phishing_websites_{i:06d}"
        for i in range(len(output))
    ]

    # Keep only rows with usable URL and label
    output = output[
        (output["url"].notna()) &
        (output["url"].str.strip() != "") &
        (output["label"].notna())
    ].copy()

    output["label"] = output["label"].astype(int)

    output = output[
        [
            "sample_id",
            "source_dataset",
            "url",
            "label",
        ]
    ]

    output.to_csv(
        PROCESSED_DIR / "uci_phishing_websites_normalized.csv",
        index=False
    )

    print(f"Saved: {PROCESSED_DIR / 'uci_phishing_websites_normalized.csv'}")
    print(f"Rows : {len(output):,}")
    print("\nLabel distribution:")
    print(output["label"].value_counts().sort_index())


def normalize_phiusiil():
    """Normalize PhiUSIIL dataset."""

    input_file = RAW_DIR / "phiusiil.csv"

    if not input_file.exists():
        print(f"Missing file: {input_file}")
        return

    print("\n" + "=" * 80)
    print("PROCESSING PHIUSIIL")
    print("=" * 80)

    df = pd.read_csv(input_file)

    url_column = find_url_column(df)
    label_column = find_label_column(df)

    print(f"Detected URL column   : {url_column}")
    print(f"Detected label column : {label_column}")

    if url_column is None or label_column is None:
        print("\nERROR: Could not automatically identify URL/label columns.")
        print("Available columns:")
        print(list(df.columns))
        return

    output = pd.DataFrame()

    output["url"] = df[url_column].astype(str).str.strip()

    # IMPORTANT:
    # PhiUSIIL uses:
    # 1 = legitimate
    # 0 = phishing
    #
    # Project convention:
    # 0 = legitimate
    # 1 = phishing
    #
    # Therefore we invert the original value.

    def convert_label(value):
        value_str = str(value).strip().lower()

        if value_str in {"1", "1.0"}:
            return 0

        if value_str in {"0", "0.0"}:
            return 1

        if value_str in {"legitimate", "benign", "safe"}:
            return 0

        if value_str in {"phishing", "phish", "malicious"}:
            return 1

        return None

    output["label"] = df[label_column].apply(convert_label)

    output["source_dataset"] = "phiusiil"

    output["sample_id"] = [
        f"phiusiil_{i:06d}"
        for i in range(len(output))
    ]

    output = output[
        (output["url"].notna()) &
        (output["url"].str.strip() != "") &
        (output["label"].notna())
    ].copy()

    output["label"] = output["label"].astype(int)

    output = output[
        [
            "sample_id",
            "source_dataset",
            "url",
            "label",
        ]
    ]

    output.to_csv(
        PROCESSED_DIR / "phiusiil_normalized.csv",
        index=False
    )

    print(f"Saved: {PROCESSED_DIR / 'phiusiil_normalized.csv'}")
    print(f"Rows : {len(output):,}")
    print("\nLabel distribution:")
    print(output["label"].value_counts().sort_index())


def main():
    print("=" * 80)
    print("URL DATASET NORMALIZATION")
    print("=" * 80)

    normalize_uci_phishing()
    normalize_phiusiil()

    print("\n" + "=" * 80)
    print("URL DATASET NORMALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()