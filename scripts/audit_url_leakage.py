from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/interim/url")
OUTPUT_DIR = Path("data/metadata")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    # Keep only information required for leakage analysis
    return df[
        [
            "sample_id",
            "source_dataset",
            "clean_url",
            "domain",
            "label",
        ]
    ].copy()


def main():

    print("=" * 80)
    print("CROSS-DATASET URL LEAKAGE AUDIT")
    print("=" * 80)

    files = sorted(INPUT_DIR.glob("*_cleaned.csv"))

    if len(files) < 2:
        print("At least two cleaned URL datasets are required.")
        return

    datasets = []

    for file_path in files:

        print(f"\nLoading: {file_path.name}")

        df = load_dataset(file_path)

        print(f"Rows: {len(df):,}")

        datasets.append(df)

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    # =========================================================
    # 1. Exact URL duplicates across datasets
    # =========================================================

    print("\n" + "=" * 80)
    print("1. EXACT URL OVERLAP")
    print("=" * 80)

    url_sources = (
        combined
        .groupby("clean_url")["source_dataset"]
        .nunique()
    )

    cross_dataset_urls = url_sources[
        url_sources > 1
    ]

    print(
        f"Unique URLs appearing in multiple datasets: "
        f"{len(cross_dataset_urls):,}"
    )

    exact_overlap_records = combined[
        combined["clean_url"].isin(
            cross_dataset_urls.index
        )
    ].copy()

    exact_overlap_file = (
        OUTPUT_DIR /
        "url_cross_dataset_exact_overlap.csv"
    )

    exact_overlap_records.to_csv(
        exact_overlap_file,
        index=False
    )

    print(
        f"Saved: {exact_overlap_file}"
    )

    # =========================================================
    # 2. Domain overlap
    # =========================================================

    print("\n" + "=" * 80)
    print("2. DOMAIN OVERLAP")
    print("=" * 80)

    domain_sources = (
        combined
        .groupby("domain")["source_dataset"]
        .nunique()
    )

    cross_dataset_domains = domain_sources[
        domain_sources > 1
    ]

    print(
        f"Domains appearing in multiple datasets: "
        f"{len(cross_dataset_domains):,}"
    )

    domain_overlap_records = combined[
        combined["domain"].isin(
            cross_dataset_domains.index
        )
    ].copy()

    domain_overlap_file = (
        OUTPUT_DIR /
        "url_cross_dataset_domain_overlap.csv"
    )

    domain_overlap_records.to_csv(
        domain_overlap_file,
        index=False
    )

    print(
        f"Saved: {domain_overlap_file}"
    )

    # =========================================================
    # 3. Conflicting labels for identical URLs
    # =========================================================

    print("\n" + "=" * 80)
    print("3. CONFLICTING LABELS")
    print("=" * 80)

    url_label_counts = (
        combined
        .groupby("clean_url")["label"]
        .nunique()
    )

    conflicting_urls = url_label_counts[
        url_label_counts > 1
    ]

    print(
        f"URLs with conflicting labels: "
        f"{len(conflicting_urls):,}"
    )

    conflicting_records = combined[
        combined["clean_url"].isin(
            conflicting_urls.index
        )
    ].copy()

    conflicting_file = (
        OUTPUT_DIR /
        "url_conflicting_labels.csv"
    )

    conflicting_records.to_csv(
        conflicting_file,
        index=False
    )

    print(
        f"Saved: {conflicting_file}"
    )

    # =========================================================
    # 4. Dataset pair overlap matrix
    # =========================================================

    print("\n" + "=" * 80)
    print("4. DATASET PAIRWISE URL OVERLAP")
    print("=" * 80)

    dataset_names = sorted(
        combined["source_dataset"].unique()
    )

    overlap_rows = []

    for dataset_a in dataset_names:

        urls_a = set(
            combined.loc[
                combined["source_dataset"] == dataset_a,
                "clean_url"
            ]
        )

        for dataset_b in dataset_names:

            if dataset_a >= dataset_b:
                continue

            urls_b = set(
                combined.loc[
                    combined["source_dataset"] == dataset_b,
                    "clean_url"
                ]
            )

            overlap = urls_a.intersection(urls_b)

            overlap_rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "urls_dataset_a": len(urls_a),
                    "urls_dataset_b": len(urls_b),
                    "shared_urls": len(overlap),
                }
            )

    overlap_df = pd.DataFrame(
        overlap_rows
    )

    pairwise_file = (
        OUTPUT_DIR /
        "url_pairwise_overlap.csv"
    )

    overlap_df.to_csv(
        pairwise_file,
        index=False
    )

    print(overlap_df.to_string(index=False))

    print(
        f"\nSaved: {pairwise_file}"
    )

    # =========================================================
    # 5. Summary report
    # =========================================================

    summary = pd.DataFrame(
        [
            {
                "total_records": len(combined),
                "unique_clean_urls": combined[
                    "clean_url"
                ].nunique(),
                "unique_domains": combined[
                    "domain"
                ].nunique(),
                "cross_dataset_shared_urls":
                    len(cross_dataset_urls),
                "cross_dataset_shared_domains":
                    len(cross_dataset_domains),
                "conflicting_label_urls":
                    len(conflicting_urls),
            }
        ]
    )

    summary_file = (
        OUTPUT_DIR /
        "url_leakage_audit_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("LEAKAGE AUDIT COMPLETE")
    print("=" * 80)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nReports saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()