from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/url/uci_phishing_websites_features.csv"
)

FEATURE_COLUMNS = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "sslfinal_state",
    "domain_registration_length",
    "favicon",
    "port",
    "https_token",
    "request_url",
    "url_of_anchor",
    "links_in_tags",
    "sfh",
    "submitting_to_email",
    "abnormal_url",
    "redirect",
    "on_mouseover",
    "rightclick",
    "popupwindow",
    "iframe",
    "age_of_domain",
    "dnsrecord",
    "web_traffic",
    "page_rank",
    "google_index",
    "links_pointing_to_page",
    "statistical_report",
]

REQUIRED_COLUMNS = (
    ["sample_id", "source_dataset"]
    + FEATURE_COLUMNS
    + ["label"]
)


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 80)
    print("UCI PHISHING WEBSITES FEATURE VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Check file exists
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    print(f"\nInput file: {INPUT_FILE}")

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # 3. Check required columns
    # --------------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("\nFAILED: Missing required columns:")
        for col in missing_columns:
            print(f"  - {col}")
        return

    print("\n[PASS] All required columns are present.")

    # --------------------------------------------------------
    # 4. Check missing values
    # --------------------------------------------------------

    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    total_missing = int(missing_values.sum())

    if total_missing > 0:
        print(
            f"\nFAILED: {total_missing:,} missing values found."
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )

        return

    print("[PASS] No missing values.")

    # --------------------------------------------------------
    # 5. Check labels
    # --------------------------------------------------------

    unique_labels = sorted(
        df["label"].dropna().unique().tolist()
    )

    print(f"\nUnique labels: {unique_labels}")

    if set(unique_labels) != {0, 1}:
        print(
            "\nFAILED: Labels must contain exactly {0, 1}."
        )
        return

    print("[PASS] Labels are exactly {0, 1}.")

    # --------------------------------------------------------
    # 6. Label distribution
    # --------------------------------------------------------

    print("\nLabel distribution:")

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in label_counts.items():

        percentage = (
            count / len(df)
        ) * 100

        label_name = (
            "Legitimate"
            if label == 0
            else "Phishing"
        )

        print(
            f"  {label} ({label_name}): "
            f"{count:,} ({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 7. Check duplicate complete rows
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        f"\nDuplicate complete rows: "
        f"{duplicate_rows:,}"
    )

    if duplicate_rows == 0:
        print("[PASS] No duplicate complete rows.")
    else:
        print(
            "[WARNING] Duplicate complete rows detected."
        )

    # --------------------------------------------------------
    # 8. Check duplicate feature vectors
    # --------------------------------------------------------

    duplicate_features = int(
        df.duplicated(
            subset=FEATURE_COLUMNS
        ).sum()
    )

    print(
        f"Duplicate feature vectors: "
        f"{duplicate_features:,}"
    )

    if duplicate_features == 0:
        print(
            "[PASS] No duplicate feature vectors."
        )
    else:
        print(
            "[WARNING] Duplicate feature vectors detected."
        )

    # --------------------------------------------------------
    # 9. Check feature values
    # --------------------------------------------------------

    print("\nFeature value ranges:")

    invalid_features = []

    for feature in FEATURE_COLUMNS:

        values = df[feature]

        min_value = values.min()
        max_value = values.max()

        print(
            f"  {feature:30s} "
            f"min={min_value:6} "
            f"max={max_value:6}"
        )

    # --------------------------------------------------------
    # 10. Final result
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("UCI FEATURE VALIDATION COMPLETE")
    print("=" * 80)

    if total_missing == 0 and set(unique_labels) == {0, 1}:
        print("\nSTATUS: PASS")
    else:
        print("\nSTATUS: FAIL")


if __name__ == "__main__":
    main()