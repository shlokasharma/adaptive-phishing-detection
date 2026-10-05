from pathlib import Path
import requests


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMAIL_DIR = PROJECT_ROOT / "data" / "raw" / "email"
URL_DIR = PROJECT_ROOT / "data" / "raw" / "url"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "raw" / "benchmarks"


# ============================================================
# Dataset sources
# ============================================================

ZENODO_RECORD_API = "https://zenodo.org/api/records/8339691"

EMAIL_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nazario_5.csv",
    "Nigerian_5.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
    "TREC_05.csv",
    "TREC_06.csv",
    "TREC_07.csv",
]

UCI_DATASETS = {
    "uci_phishing_websites": {
        "url": "https://archive.ics.uci.edu/static/public/327/data.zip",
        "filename": "uci_phishing_websites.zip",
    },
    "phiusiil": {
        "url": "https://archive.ics.uci.edu/static/public/967/data.zip",
        "filename": "phiusiil.zip",
    },
}


# ============================================================
# Utility functions
# ============================================================

def download_file(url: str, destination: Path) -> bool:
    """
    Download a file unless it already exists.

    Returns:
        True  -> downloaded
        False -> skipped
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"[SKIP] {destination.name} already exists.")
        return False

    print(f"\n[DOWNLOAD] {destination.name}")
    print(f"Source: {url}")

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=120,
            headers={"User-Agent": "adaptive-phishing-detection-project"},
        )

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percentage = downloaded / total_size * 100
                    print(
                        f"\rProgress: {percentage:6.2f}%",
                        end="",
                    )

        print("\n[OK] Download completed.")
        return True

    except requests.RequestException as error:
        print(f"\n[ERROR] Download failed: {error}")

        if destination.exists():
            destination.unlink()

        return False


# ============================================================
# Zenodo email datasets
# ============================================================

def download_email_datasets():
    print("\n" + "=" * 60)
    print("EMAIL DATASETS")
    print("=" * 60)

    EMAIL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRetrieving current Zenodo file information...")

    response = requests.get(
        ZENODO_RECORD_API,
        timeout=60,
        headers={"User-Agent": "adaptive-phishing-detection-project"},
    )

    response.raise_for_status()

    record = response.json()

    available_files = {
        item["key"]: item["links"]["self"]
        for item in record.get("files", [])
    }

    print(f"Found {len(available_files)} files in Zenodo record.")

    for filename in EMAIL_FILES:

        if filename not in available_files:
            print(f"[WARNING] {filename} not found in Zenodo record.")
            continue

        destination = EMAIL_DIR / filename

        download_file(
            available_files[filename],
            destination,
        )


# ============================================================
# UCI URL datasets
# ============================================================

def download_url_datasets():
    print("\n" + "=" * 60)
    print("URL DATASETS")
    print("=" * 60)

    URL_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, information in UCI_DATASETS.items():

        destination = URL_DIR / information["filename"]

        print(f"\nDataset: {dataset_name}")

        download_file(
            information["url"],
            destination,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("ADAPTIVE PHISHING DETECTION")
    print("MASTER DATASET ACQUISITION")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nThe following datasets will be acquired:")
    print("\nEMAIL:")
    for filename in EMAIL_FILES:
        print(f"  - {filename}")

    print("\nURL:")
    for dataset_name in UCI_DATASETS:
        print(f"  - {dataset_name}")

    print("\nExisting files will be skipped automatically.")

    # Download email datasets
    download_email_datasets()

    # Download URL datasets
    download_url_datasets()

    print("\n" + "=" * 60)
    print("DATASET ACQUISITION FINISHED")
    print("=" * 60)

    print("\nCheck:")
    print("  data/raw/email/")
    print("  data/raw/url/")

    print("\nIMPORTANT:")
    print("Raw datasets have NOT been modified or processed.")


if __name__ == "__main__":
    main()