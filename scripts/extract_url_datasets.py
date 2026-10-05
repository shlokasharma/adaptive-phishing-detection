from pathlib import Path
import zipfile


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "url"
)


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "uci_phishing_websites.zip": "uci_phishing_websites",
    "phiusiil.zip": "phiusiil",
}


# ============================================================
# EXTRACTION FUNCTION
# ============================================================

def extract_dataset(zip_filename, output_folder):

    zip_path = URL_DIR / zip_filename
    output_path = URL_DIR / output_folder

    print("\n" + "=" * 80)
    print(f"DATASET: {zip_filename}")
    print("=" * 80)

    if not zip_path.exists():

        print(
            f"\nERROR: ZIP file not found:"
        )

        print(zip_path)

        return False

    if output_path.exists():

        print(
            "\nOutput folder already exists:"
        )

        print(output_path)

        print(
            "\nSkipping extraction."
        )

        return True

    try:

        print(
            f"\nExtracting to:"
        )

        print(output_path)

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                output_path
            )

        print(
            "\nExtraction successful."
        )

        return True

    except zipfile.BadZipFile:

        print(
            "\nERROR: The ZIP file appears "
            "to be corrupted or invalid."
        )

        return False

    except Exception as error:

        print(
            f"\nERROR during extraction: "
            f"{error}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("UCI URL DATASET EXTRACTION")
    print("=" * 80)

    successful = 0

    for zip_filename, output_folder in DATASETS.items():

        if extract_dataset(
            zip_filename,
            output_folder
        ):

            successful += 1

    print("\n" + "=" * 80)
    print("URL DATASET EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"\nSuccessfully extracted: "
        f"{successful}/{len(DATASETS)}"
    )


if __name__ == "__main__":
    main()