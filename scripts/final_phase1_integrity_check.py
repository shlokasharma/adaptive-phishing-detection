from pathlib import Path
import pandas as pd
import sys


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INTERIM = DATA / "interim"
SPLITS = DATA / "splits"
METADATA = DATA / "metadata"


# ============================================================================
# INTEGRITY TRACKING
# ============================================================================

errors = []
warnings = []


def check(condition, message):
    """
    Register a mandatory integrity check.
    """
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        errors.append(message)


def warn(message):
    """
    Register a non-blocking warning.
    """
    print(f"[WARN] {message}")
    warnings.append(message)


def load_csv(path):
    """
    Safely load a CSV file.
    """
    try:
        return pd.read_csv(path)
    except Exception as exc:
        message = f"Could not read {path}: {exc}"
        print(f"[FAIL] {message}")
        errors.append(message)
        return None


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("FINAL PHASE-1 INTEGRITY CHECK")
print("=" * 80)


# ============================================================================
# 1. REQUIRED METADATA FILES
# ============================================================================

print("\n[1] CHECKING METADATA FILES")

metadata_files = [
    METADATA / "dataset_manifest.csv",
    METADATA / "label_audit.csv",
    METADATA / "email_cross_dataset_duplicates.csv",
    METADATA / "email_leakage_summary.csv",
    METADATA / "phiusiil_leakage_summary.csv",
    METADATA / "phiusiil_mixed_label_domains.csv",
    METADATA / "dataset_registry.csv",
    METADATA / "split_manifest.csv",
    METADATA / "dataset_statistics.csv",
]

for path in metadata_files:
    check(
        path.exists(),
        f"Metadata file exists: {path.relative_to(ROOT)}"
    )


# ============================================================================
# 2. REQUIRED DATASET FILES
# ============================================================================

print("\n[2] CHECKING CORE DATASET FILES")

required_files = [
    # Raw URL datasets
    RAW / "url" / "uci_phishing_websites.csv",
    RAW / "url" / "phiusiil.csv",

    # Correct processed UCI feature dataset
    PROCESSED / "url" / "uci_phishing_websites_features.csv",

    # Cleaned PhiUSIIL URL dataset
    INTERIM / "url" / "phiusiil_cleaned.csv",

    # UCI feature splits
    SPLITS / "url" / "uci_features" / "train.csv",
    SPLITS / "url" / "uci_features" / "validation.csv",
    SPLITS / "url" / "uci_features" / "test.csv",

    # PhiUSIIL URL splits
    SPLITS / "url" / "phiusiil" / "train.csv",
    SPLITS / "url" / "phiusiil" / "validation.csv",
    SPLITS / "url" / "phiusiil" / "test.csv",

    # Email splits
    SPLITS / "email" / "train.csv",
    SPLITS / "email" / "validation.csv",
    SPLITS / "email" / "test.csv",
]

for path in required_files:
    check(
        path.exists(),
        f"Required file exists: {path.relative_to(ROOT)}"
    )


# ============================================================================
# 3. UCI PHISHING WEBSITES FEATURE DATASET
# ============================================================================

print("\n[3] CHECKING UCI PHISHING WEBSITES FEATURE DATASET")

uci_path = (
    PROCESSED
    / "url"
    / "uci_phishing_websites_features.csv"
)

uci_train_path = (
    SPLITS
    / "url"
    / "uci_features"
    / "train.csv"
)

uci_val_path = (
    SPLITS
    / "url"
    / "uci_features"
    / "validation.csv"
)

uci_test_path = (
    SPLITS
    / "url"
    / "uci_features"
    / "test.csv"
)

uci = load_csv(uci_path)
uci_train = load_csv(uci_train_path)
uci_val = load_csv(uci_val_path)
uci_test = load_csv(uci_test_path)


if all(
    df is not None
    for df in [uci, uci_train, uci_val, uci_test]
):

    # ------------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------------

    check(
        set(uci["label"].unique()) == {0, 1},
        "UCI labels are exactly {0, 1}"
    )

    # ------------------------------------------------------------------------
    # Sample IDs
    # ------------------------------------------------------------------------

    check(
        uci["sample_id"].is_unique,
        "UCI sample IDs are unique"
    )

    # ------------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------------

    check(
        not uci.isnull().any().any(),
        "UCI feature dataset contains no missing values"
    )

    # ------------------------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------------------------

    feature_columns = [
        column
        for column in uci.columns
        if column not in [
            "sample_id",
            "source_dataset",
            "label",
        ]
    ]

    check(
        len(feature_columns) == 30,
        f"UCI contains 30 feature columns (found {len(feature_columns)})"
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The UCI feature split pipeline removes duplicate feature vectors
    # BEFORE splitting.
    #
    # Therefore:
    #
    # processed rows != split rows
    #
    # The correct integrity condition is:
    #
    # split rows == number of unique feature vectors
    # ------------------------------------------------------------------------

    unique_uci_vectors = uci[
        feature_columns
    ].drop_duplicates()

    total_split_rows = (
        len(uci_train)
        + len(uci_val)
        + len(uci_test)
    )

    check(
        total_split_rows == len(unique_uci_vectors),
        (
            "UCI split row counts match the "
            "deduplicated feature-vector dataset"
        )
    )

    # ------------------------------------------------------------------------
    # Sample-ID overlap
    # ------------------------------------------------------------------------

    train_ids = set(uci_train["sample_id"])
    val_ids = set(uci_val["sample_id"])
    test_ids = set(uci_test["sample_id"])

    check(
        len(train_ids & val_ids) == 0,
        "UCI train/validation sample IDs are disjoint"
    )

    check(
        len(train_ids & test_ids) == 0,
        "UCI train/test sample IDs are disjoint"
    )

    check(
        len(val_ids & test_ids) == 0,
        "UCI validation/test sample IDs are disjoint"
    )

    # ------------------------------------------------------------------------
    # Feature-vector overlap
    # ------------------------------------------------------------------------

    train_vectors = set(
        map(
            tuple,
            uci_train[feature_columns].to_numpy()
        )
    )

    val_vectors = set(
        map(
            tuple,
            uci_val[feature_columns].to_numpy()
        )
    )

    test_vectors = set(
        map(
            tuple,
            uci_test[feature_columns].to_numpy()
        )
    )

    check(
        len(train_vectors & val_vectors) == 0,
        "UCI train/validation feature vectors are disjoint"
    )

    check(
        len(train_vectors & test_vectors) == 0,
        "UCI train/test feature vectors are disjoint"
    )

    check(
        len(val_vectors & test_vectors) == 0,
        "UCI validation/test feature vectors are disjoint"
    )


# ============================================================================
# 4. PHIUSIIL URL DATASET
# ============================================================================

print("\n[4] CHECKING PHIUSIIL URL DATASET")

phi_path = (
    INTERIM
    / "url"
    / "phiusiil_cleaned.csv"
)

phi_train_path = (
    SPLITS
    / "url"
    / "phiusiil"
    / "train.csv"
)

phi_val_path = (
    SPLITS
    / "url"
    / "phiusiil"
    / "validation.csv"
)

phi_test_path = (
    SPLITS
    / "url"
    / "phiusiil"
    / "test.csv"
)

phi = load_csv(phi_path)
phi_train = load_csv(phi_train_path)
phi_val = load_csv(phi_val_path)
phi_test = load_csv(phi_test_path)


if all(
    df is not None
    for df in [phi, phi_train, phi_val, phi_test]
):

    # ------------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------------

    check(
        set(phi["label"].unique()) == {0, 1},
        "PhiUSIIL labels are exactly {0, 1}"
    )

    # ------------------------------------------------------------------------
    # Sample IDs
    # ------------------------------------------------------------------------

    check(
        phi["sample_id"].is_unique,
        "PhiUSIIL sample IDs are unique after cleaning"
    )

    # ------------------------------------------------------------------------
    # Required URL fields
    # ------------------------------------------------------------------------

    check(
        "domain" in phi.columns,
        "PhiUSIIL contains domain information"
    )

    check(
        "clean_url" in phi.columns,
        "PhiUSIIL contains canonicalized clean URLs"
    )

    # ------------------------------------------------------------------------
    # Sample-ID overlap
    # ------------------------------------------------------------------------

    train_ids = set(phi_train["sample_id"])
    val_ids = set(phi_val["sample_id"])
    test_ids = set(phi_test["sample_id"])

    check(
        len(train_ids & val_ids) == 0,
        "PhiUSIIL train/validation sample IDs are disjoint"
    )

    check(
        len(train_ids & test_ids) == 0,
        "PhiUSIIL train/test sample IDs are disjoint"
    )

    check(
        len(val_ids & test_ids) == 0,
        "PhiUSIIL validation/test sample IDs are disjoint"
    )

    # ------------------------------------------------------------------------
    # Domain overlap
    # ------------------------------------------------------------------------

    train_domains = set(
        phi_train["domain"].dropna()
    )

    val_domains = set(
        phi_val["domain"].dropna()
    )

    test_domains = set(
        phi_test["domain"].dropna()
    )

    check(
        len(train_domains & val_domains) == 0,
        "PhiUSIIL train/validation domains are disjoint"
    )

    check(
        len(train_domains & test_domains) == 0,
        "PhiUSIIL train/test domains are disjoint"
    )

    check(
        len(val_domains & test_domains) == 0,
        "PhiUSIIL validation/test domains are disjoint"
    )

    # ------------------------------------------------------------------------
    # URL overlap
    # ------------------------------------------------------------------------

    all_split_urls = pd.concat(
        [
            phi_train["clean_url"],
            phi_val["clean_url"],
            phi_test["clean_url"],
        ],
        ignore_index=True,
    )

    check(
        all_split_urls.is_unique,
        "PhiUSIIL clean URLs are unique across all splits"
    )


# ============================================================================
# 5. EMAIL DATASET
# ============================================================================

print("\n[5] CHECKING EMAIL DATASET")

email_dir = INTERIM / "email"

email_files = sorted(
    email_dir.glob("*_cleaned.csv")
)

check(
    len(email_files) == 9,
    (
        "Nine readable email datasets are present "
        f"(found {len(email_files)})"
    )
)

email_frames = []

for path in email_files:

    df = load_csv(path)

    if df is not None:
        email_frames.append(df)


if email_frames:

    email_all = pd.concat(
        email_frames,
        ignore_index=True
    )

    # ------------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------------

    check(
        "label" in email_all.columns,
        "Email datasets contain label column"
    )

    check(
        "sample_id" in email_all.columns,
        "Email datasets contain sample_id column"
    )

    check(
        "clean_text" in email_all.columns,
        "Email datasets contain clean_text column"
    )

    # ------------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------------

    check(
        set(email_all["label"].dropna().unique()) <= {0, 1},
        "Email labels use only 0/1"
    )

    # ------------------------------------------------------------------------
    # Sample IDs
    # ------------------------------------------------------------------------

    check(
        email_all["sample_id"].is_unique,
        "Email sample IDs are globally unique"
    )

    # ------------------------------------------------------------------------
    # Text completeness
    # ------------------------------------------------------------------------

    check(
        email_all["clean_text"].notna().all(),
        "Email clean_text contains no missing values"
    )

    # ------------------------------------------------------------------------
    # Email splits
    # ------------------------------------------------------------------------

    email_train_path = (
        SPLITS
        / "email"
        / "train.csv"
    )

    email_val_path = (
        SPLITS
        / "email"
        / "validation.csv"
    )

    email_test_path = (
        SPLITS
        / "email"
        / "test.csv"
    )

    email_train = load_csv(email_train_path)
    email_val = load_csv(email_val_path)
    email_test = load_csv(email_test_path)

    if all(
        df is not None
        for df in [
            email_train,
            email_val,
            email_test,
        ]
    ):

        # --------------------------------------------------------------------
        # Sample-ID overlap
        # --------------------------------------------------------------------

        train_ids = set(
            email_train["sample_id"]
        )

        val_ids = set(
            email_val["sample_id"]
        )

        test_ids = set(
            email_test["sample_id"]
        )

        check(
            len(train_ids & val_ids) == 0,
            "Email train/validation sample IDs are disjoint"
        )

        check(
            len(train_ids & test_ids) == 0,
            "Email train/test sample IDs are disjoint"
        )

        check(
            len(val_ids & test_ids) == 0,
            "Email validation/test sample IDs are disjoint"
        )

        # --------------------------------------------------------------------
        # Text overlap
        # --------------------------------------------------------------------

        train_text = set(
            email_train["clean_text"]
        )

        val_text = set(
            email_val["clean_text"]
        )

        test_text = set(
            email_test["clean_text"]
        )

        check(
            len(train_text & val_text) == 0,
            "Email train/validation clean text is disjoint"
        )

        check(
            len(train_text & test_text) == 0,
            "Email train/test clean text is disjoint"
        )

        check(
            len(val_text & test_text) == 0,
            "Email validation/test clean text is disjoint"
        )

        # --------------------------------------------------------------------
        # Row-count consistency
        # --------------------------------------------------------------------

        total_email_split_rows = (
            len(email_train)
            + len(email_val)
            + len(email_test)
        )

        check(
            total_email_split_rows == len(email_all),
            (
                "Email split row counts match "
                "the combined cleaned datasets"
            )
        )


# ============================================================================
# 6. CHECK FOR OLD INVALID UCI ARTIFACTS
# ============================================================================

print("\n[6] CHECKING FOR INVALID UCI URL ARTIFACTS")

old_uci_files = [
    PROCESSED
    / "url"
    / "uci_phishing_websites_normalized.csv",

    INTERIM
    / "url"
    / "uci_phishing_websites_cleaned.csv",
]

for path in old_uci_files:

    check(
        not path.exists(),
        (
            "Old invalid UCI URL artifact removed: "
            f"{path.relative_to(ROOT)}"
        )
    )


# ============================================================================
# 7. KNOWN EMAIL DATASET EXCEPTIONS
# ============================================================================

print("\n[7] CHECKING KNOWN EMAIL DATASET EXCEPTIONS")

trec05 = RAW / "email" / "TREC_05.csv"
trec06 = RAW / "email" / "TREC_06.csv"


if trec05.exists():

    warn(
        "TREC_05.csv is present but was previously "
        "unreadable during inspection."
    )

else:

    warn(
        "TREC_05.csv is not present in the current "
        "raw email directory."
    )


if trec06.exists():

    warn(
        "TREC_06.csv is present but was previously "
        "unreadable during inspection."
    )

else:

    warn(
        "TREC_06.csv is not present in the current "
        "raw email directory."
    )


# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nTotal checks failed : {len(errors)}")
print(f"Warnings            : {len(warnings)}")


if errors:

    print("\nPHASE 1 INTEGRITY STATUS: FAIL")

    print("\nFailures:")

    for error in errors:
        print(f"  - {error}")

    print(
        "\nPlease fix the failures before "
        "freezing Phase 1."
    )

    print("=" * 80)

    sys.exit(1)


else:

    print("\nPHASE 1 INTEGRITY STATUS: PASS")

    if warnings:

        print(
            "\nWarnings that are documented "
            "but do not block Phase 1:"
        )

        for warning in warnings:
            print(f"  - {warning}")

    print(
        "\nAll mandatory Phase-1 integrity "
        "checks passed."
    )

    print(
        "Phase 1 is ready for final freeze documentation."
    )

    print("=" * 80)