from pathlib import Path
import pandas as pd


INPUT_FILE = Path(
    "data/interim/url/phiusiil_cleaned.csv"
)

OUTPUT_DIR = Path("data/metadata")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 80)
    print("PHIUSIIL URL LEAKAGE AUDIT")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "sample_id",
        "source_dataset",
        "original_url",
        "clean_url",
        "domain",
        "label",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("\n[PASS] Required columns found.")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing_counts = df[required].isna().sum()

    print(missing_counts)

    # --------------------------------------------------------
    # Duplicate URLs
    # --------------------------------------------------------

    duplicate_urls = int(
        df.duplicated(
            subset=["clean_url"]
        ).sum()
    )

    unique_urls = (
        df["clean_url"]
        .nunique()
    )

    print("\nURL statistics:")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique URLs      : {unique_urls:,}")
    print(f"  Duplicate URLs   : {duplicate_urls:,}")

    # --------------------------------------------------------
    # Duplicate domains
    # --------------------------------------------------------

    unique_domains = (
        df["domain"]
        .nunique()
    )

    print("\nDomain statistics:")
    print(f"  Unique domains   : {unique_domains:,}")

    # --------------------------------------------------------
    # Conflicting labels for same URL
    # --------------------------------------------------------

    label_counts = (
        df.groupby("clean_url")["label"]
        .nunique()
    )

    conflicting_urls = label_counts[
        label_counts > 1
    ]

    print(
        "\nConflicting-label URLs: "
        f"{len(conflicting_urls):,}"
    )

    if len(conflicting_urls) > 0:

        conflict_df = (
            df[
                df["clean_url"].isin(
                    conflicting_urls.index
                )
            ]
            .sort_values("clean_url")
        )

        conflict_path = (
            OUTPUT_DIR /
            "phiusiil_conflicting_urls.csv"
        )

        conflict_df.to_csv(
            conflict_path,
            index=False
        )

        print(
            f"Conflict report saved to: "
            f"{conflict_path}"
        )

    # --------------------------------------------------------
    # Domain-label analysis
    # --------------------------------------------------------

    domain_label_counts = (
        df.groupby("domain")["label"]
        .nunique()
    )

    mixed_domains = domain_label_counts[
        domain_label_counts > 1
    ]

    print(
        "\nDomains containing both labels: "
        f"{len(mixed_domains):,}"
    )

    if len(mixed_domains) > 0:

        mixed_domain_df = (
            df[
                df["domain"].isin(
                    mixed_domains.index
                )
            ]
            .sort_values("domain")
        )

        mixed_domain_path = (
            OUTPUT_DIR /
            "phiusiil_mixed_label_domains.csv"
        )

        mixed_domain_df.to_csv(
            mixed_domain_path,
            index=False
        )

        print(
            f"Mixed-domain report saved to: "
            f"{mixed_domain_path}"
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print("\nLabel distribution:")

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in label_counts.items():

        label_name = (
            "Legitimate"
            if label == 0
            else "Phishing"
        )

        print(
            f"  {label} ({label_name}): "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "unique_urls",
                "duplicate_urls",
                "unique_domains",
                "conflicting_label_urls",
                "mixed_label_domains",
            ],
            "value": [
                len(df),
                unique_urls,
                duplicate_urls,
                unique_domains,
                len(conflicting_urls),
                len(mixed_domains),
            ],
        }
    )

    summary_path = (
        OUTPUT_DIR /
        "phiusiil_leakage_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        f"\nSummary saved to: {summary_path}"
    )

    print("\n" + "=" * 80)
    print("PHIUSIIL LEAKAGE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()