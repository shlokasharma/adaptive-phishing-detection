from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_DIR = Path("data/metadata")

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET DEFINITIONS
# ============================================================

datasets = [
    {
        "dataset_id": "uci_phishing_websites",
        "dataset_name": "UCI Phishing Websites",
        "modality": "url_features",
        "role": "core",
        "raw_path": "data/raw/url/uci_phishing_websites.csv",
        "processed_path": (
            "data/processed/url/"
            "uci_phishing_websites_features.csv"
        ),
        "split_path": "data/splits/url/uci_features/",
        "label_definition": (
            "0 = legitimate, 1 = phishing"
        ),
        "split_strategy": (
            "Stratified sample split after duplicate "
            "feature-vector removal"
        ),
        "notes": (
            "Feature-based dataset; does not contain "
            "actual raw URLs."
        ),
    },
    {
        "dataset_id": "phiusiil",
        "dataset_name": "PhiUSIIL Phishing URL Dataset",
        "modality": "url",
        "role": "core",
        "raw_path": "data/raw/url/phiusiil.csv",
        "processed_path": (
            "data/interim/url/phiusiil_cleaned.csv"
        ),
        "split_path": "data/splits/url/phiusiil/",
        "label_definition": (
            "0 = legitimate, 1 = phishing"
        ),
        "split_strategy": (
            "Domain-disjoint split; identical URLs "
            "removed before splitting"
        ),
        "notes": (
            "Contains actual URLs and extracted URL/domain "
            "information."
        ),
    },
    {
        "dataset_id": "email_combined",
        "dataset_name": "Email Dataset Collection",
        "modality": "email",
        "role": "core",
        "raw_path": "data/raw/email/",
        "processed_path": "data/interim/email/",
        "split_path": "data/splits/email/",
        "label_definition": (
            "0 = legitimate, 1 = phishing/spam/fraud "
            "according to source normalization"
        ),
        "split_strategy": (
            "Text-group-disjoint split; identical cleaned "
            "email texts remain in one split"
        ),
        "notes": (
            "Multiple public email sources retained with "
            "source_dataset identity."
        ),
    },
]


# ============================================================
# FILE EXISTENCE CHECK
# ============================================================

def path_status(path_string):

    path = Path(path_string)

    if path.exists():
        return "exists"

    return "missing"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DATASET METADATA CONSOLIDATION")
    print("=" * 80)

    metadata_rows = []

    for dataset in datasets:

        row = dataset.copy()

        row["raw_status"] = path_status(
            dataset["raw_path"]
        )

        row["processed_status"] = path_status(
            dataset["processed_path"]
        )

        row["split_status"] = path_status(
            dataset["split_path"]
        )

        metadata_rows.append(row)

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    # --------------------------------------------------------
    # Save master dataset registry
    # --------------------------------------------------------

    registry_path = (
        METADATA_DIR /
        "dataset_registry.csv"
    )

    metadata_df.to_csv(
        registry_path,
        index=False
    )

    print(
        f"\nDataset registry saved to:"
        f"\n  {registry_path}"
    )

    # --------------------------------------------------------
    # Create split manifest
    # --------------------------------------------------------

    split_files = [
        Path(
            "data/splits/url/uci_features/train.csv"
        ),
        Path(
            "data/splits/url/uci_features/validation.csv"
        ),
        Path(
            "data/splits/url/uci_features/test.csv"
        ),
        Path(
            "data/splits/url/phiusiil/train.csv"
        ),
        Path(
            "data/splits/url/phiusiil/validation.csv"
        ),
        Path(
            "data/splits/url/phiusiil/test.csv"
        ),
        Path(
            "data/splits/email/train.csv"
        ),
        Path(
            "data/splits/email/validation.csv"
        ),
        Path(
            "data/splits/email/test.csv"
        ),
    ]

    split_rows = []

    for path in split_files:

        row = {
            "split_file": str(path),
            "exists": path.exists(),
        }

        if path.exists():

            try:

                df = pd.read_csv(path)

                row["rows"] = len(df)

                row["columns"] = len(df.columns)

                row["label_0"] = int(
                    (df["label"] == 0).sum()
                )

                row["label_1"] = int(
                    (df["label"] == 1).sum()
                )

            except Exception as exc:

                row["rows"] = None
                row["columns"] = None
                row["label_0"] = None
                row["label_1"] = None
                row["error"] = str(exc)

        else:

            row["rows"] = None
            row["columns"] = None
            row["label_0"] = None
            row["label_1"] = None

        split_rows.append(row)

    split_manifest = pd.DataFrame(
        split_rows
    )

    split_manifest_path = (
        METADATA_DIR /
        "split_manifest.csv"
    )

    split_manifest.to_csv(
        split_manifest_path,
        index=False
    )

    print(
        f"\nSplit manifest saved to:"
        f"\n  {split_manifest_path}"
    )

    # --------------------------------------------------------
    # Create dataset statistics
    # --------------------------------------------------------

    statistics = []

    # UCI
    uci_path = Path(
        "data/processed/url/"
        "uci_phishing_websites_features.csv"
    )

    if uci_path.exists():

        uci = pd.read_csv(
            uci_path
        )

        statistics.append(
            {
                "dataset_id": "uci_phishing_websites",
                "rows": len(uci),
                "columns": len(uci.columns),
                "label_0": int(
                    (uci["label"] == 0).sum()
                ),
                "label_1": int(
                    (uci["label"] == 1).sum()
                ),
            }
        )

    # PhiUSIIL
    phi_path = Path(
        "data/interim/url/"
        "phiusiil_cleaned.csv"
    )

    if phi_path.exists():

        phi = pd.read_csv(
            phi_path
        )

        statistics.append(
            {
                "dataset_id": "phiusiil",
                "rows": len(phi),
                "columns": len(phi.columns),
                "label_0": int(
                    (phi["label"] == 0).sum()
                ),
                "label_1": int(
                    (phi["label"] == 1).sum()
                ),
            }
        )

    # Email
    email_files = sorted(
        Path("data/interim/email")
        .glob("*_cleaned.csv")
    )

    if email_files:

        email_frames = [
            pd.read_csv(file)
            for file in email_files
        ]

        email = pd.concat(
            email_frames,
            ignore_index=True
        )

        statistics.append(
            {
                "dataset_id": "email_combined",
                "rows": len(email),
                "columns": len(email.columns),
                "label_0": int(
                    (email["label"] == 0).sum()
                ),
                "label_1": int(
                    (email["label"] == 1).sum()
                ),
            }
        )

    statistics_df = pd.DataFrame(
        statistics
    )

    statistics_path = (
        METADATA_DIR /
        "dataset_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_path,
        index=False
    )

    print(
        f"\nDataset statistics saved to:"
        f"\n  {statistics_path}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATASET METADATA CONSOLIDATION COMPLETE")
    print("=" * 80)

    print(
        "\nMetadata files created:"
    )

    print(
        f"  {registry_path}"
    )

    print(
        f"  {split_manifest_path}"
    )

    print(
        f"  {statistics_path}"
    )


if __name__ == "__main__":
    main()