from pathlib import Path

import pandas as pd


RAW_FILE = Path(
    "data/raw/url/uci_phishing_websites.csv"
)

OUTPUT_DIR = Path(
    "data/processed/url"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EXPECTED_FEATURES = [
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


def main():

    print("=" * 80)
    print("UCI PHISHING WEBSITES FEATURE PROCESSING")
    print("=" * 80)

    if not RAW_FILE.exists():

        print(
            f"ERROR: Dataset not found:\n{RAW_FILE}"
        )

        return

    # ---------------------------------------------------------
    # Load raw dataset
    # ---------------------------------------------------------

    df = pd.read_csv(RAW_FILE)

    print(f"\nRows loaded    : {len(df):,}")
    print(f"Columns loaded : {len(df.columns)}")

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    missing_features = [
        column
        for column in EXPECTED_FEATURES
        if column not in df.columns
    ]

    if missing_features:

        print("\nERROR: Missing expected features:")

        for column in missing_features:
            print(f"  - {column}")

        return

    if "result" not in df.columns:

        print(
            "\nERROR: Target column 'result' "
            "was not found."
        )

        return

    print("\nAll expected feature columns found.")

    # ---------------------------------------------------------
    # Create output
    # ---------------------------------------------------------

    output = df[
        EXPECTED_FEATURES + ["result"]
    ].copy()

    # ---------------------------------------------------------
    # Remove missing rows
    # ---------------------------------------------------------

    before = len(output)

    output = output.dropna().copy()

    removed_missing = before - len(output)

    print(
        f"\nRows removed because of missing values: "
        f"{removed_missing:,}"
    )

    # ---------------------------------------------------------
    # Convert target
    # ---------------------------------------------------------
    #
    # UCI Phishing Websites:
    #
    # result = -1 → legitimate
    # result =  1 → phishing
    #
    # Project convention:
    #
    # label = 0 → legitimate
    # label = 1 → phishing
    #
    # ---------------------------------------------------------

    def convert_label(value):

        if value == -1:
            return 0

        if value == 1:
            return 1

        return None

    output["label"] = (
        output["result"]
        .apply(convert_label)
    )

    invalid_labels = output["label"].isna().sum()

    print(
        f"Invalid target values: "
        f"{invalid_labels:,}"
    )

    if invalid_labels > 0:

        print(
            "WARNING: Invalid target rows "
            "will be removed."
        )

        output = output[
            output["label"].notna()
        ].copy()

    output["label"] = output["label"].astype(int)

    # ---------------------------------------------------------
    # Add sample IDs
    # ---------------------------------------------------------

    output.insert(
        0,
        "sample_id",
        [
            f"uci_phishing_features_{i:06d}"
            for i in range(len(output))
        ]
    )

    output.insert(
        1,
        "source_dataset",
        "uci_phishing_websites"
    )

    # ---------------------------------------------------------
    # Remove original result column
    # ---------------------------------------------------------

    output = output.drop(
        columns=["result"]
    )

    # ---------------------------------------------------------
    # Final column order
    # ---------------------------------------------------------

    output_columns = [
        "sample_id",
        "source_dataset",
    ] + EXPECTED_FEATURES + [
        "label"
    ]

    output = output[
        output_columns
    ]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "uci_phishing_websites_features.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PROCESSING RESULT")
    print("=" * 80)

    print(
        f"Final rows: {len(output):,}"
    )

    print(
        f"Final columns: {len(output.columns)}"
    )

    print("\nLabel distribution:")

    print(
        output["label"]
        .value_counts()
        .sort_index()
    )

    print("\nFeature columns:")

    for feature in EXPECTED_FEATURES:
        print(f"  - {feature}")

    print("\nSaved to:")

    print(output_file)

    print("\n" + "=" * 80)
    print("UCI FEATURE PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()