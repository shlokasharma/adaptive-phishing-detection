from pathlib import Path
import re
import html
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "email"
)

OUTPUT_EMAIL_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "email"
)

OUTPUT_URL_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "extracted"
)

OUTPUT_EMAIL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_URL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# URL REGEX
# ============================================================

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+|"
    r"www\.[^\s<>\"]+",
    re.IGNORECASE
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_email_text(text):
    """
    Clean email text while preserving the semantic content.

    Operations:
    - decode HTML entities
    - remove HTML tags
    - normalize line endings
    - normalize whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize line breaks
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# URL EXTRACTION
# ============================================================

def extract_urls(text):

    if not text:
        return []

    matches = URL_PATTERN.findall(text)

    cleaned_urls = []

    for url in matches:

        # Remove common punctuation attached to URLs
        url = url.rstrip(
            ".,;:!?)]}>\"'"
        )

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    return cleaned_urls


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(file_path):

    print("\n" + "=" * 80)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 80)

    df = pd.read_csv(
        file_path,
        low_memory=False,
        encoding_errors="replace"
    )

    print(
        f"Input records: {len(df):,}"
    )

    # --------------------------------------------------------
    # Clean subject and body
    # --------------------------------------------------------

    df["subject_clean"] = (
        df["subject"]
        .fillna("")
        .apply(clean_email_text)
    )

    df["body_clean"] = (
        df["body"]
        .fillna("")
        .apply(clean_email_text)
    )

    # --------------------------------------------------------
    # Create cleaned combined text
    # --------------------------------------------------------

    df["clean_text"] = (
        df["subject_clean"]
        + " "
        + df["body_clean"]
    ).str.strip()

    # --------------------------------------------------------
    # Extract URLs
    # --------------------------------------------------------

    df["extracted_urls"] = (
        df["clean_text"]
        .apply(extract_urls)
    )

    df["url_count"] = (
        df["extracted_urls"]
        .apply(len)
    )

    # --------------------------------------------------------
    # Create URL evidence table
    # --------------------------------------------------------

    url_records = []

    for _, row in df.iterrows():

        urls = row["extracted_urls"]

        for url in urls:

            url_records.append(
                {
                    "email_sample_id":
                        row["sample_id"],

                    "source_dataset":
                        row["source_dataset"],

                    "label":
                        row["label"],

                    "url":
                        url,
                }
            )

    url_df = pd.DataFrame(
        url_records,
        columns=[
            "email_sample_id",
            "source_dataset",
            "label",
            "url",
        ]
    )

    # --------------------------------------------------------
    # Remove URL list before saving email dataset
    # --------------------------------------------------------

    email_output = df.drop(
        columns=["extracted_urls"]
    )

    # --------------------------------------------------------
    # Save cleaned email dataset
    # --------------------------------------------------------

    email_output_path = (
        OUTPUT_EMAIL_DIR
        / file_path.name.replace(
            "_normalized.csv",
            "_cleaned.csv"
        )
    )

    email_output.to_csv(
        email_output_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save URL evidence
    # --------------------------------------------------------

    url_output_path = (
        OUTPUT_URL_DIR
        / file_path.name.replace(
            "_normalized.csv",
            "_urls.csv"
        )
    )

    url_df.to_csv(
        url_output_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    emails_with_urls = (
        (df["url_count"] > 0)
        .sum()
    )

    total_urls = (
        df["url_count"]
        .sum()
    )

    print(
        f"Emails containing URLs : "
        f"{emails_with_urls:,}"
    )

    print(
        f"URLs extracted         : "
        f"{total_urls:,}"
    )

    print(
        f"\nCleaned email file:"
    )

    print(email_output_path)

    print(
        f"\nURL evidence file:"
    )

    print(url_output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EMAIL CLEANING AND URL EXTRACTION")
    print("=" * 80)

    files = sorted(
        INPUT_DIR.glob(
            "*_normalized.csv"
        )
    )

    if not files:

        print(
            "\nNo normalized email datasets found."
        )

        return

    print(
        f"\nDatasets found: {len(files)}"
    )

    for file_path in files:

        try:

            process_dataset(
                file_path
            )

        except Exception as error:

            print(
                f"\nERROR processing "
                f"{file_path.name}"
            )

            print(error)

    print("\n" + "=" * 80)
    print("CLEANING AND URL EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"\nCleaned email output:\n"
        f"{OUTPUT_EMAIL_DIR}"
    )

    print(
        f"\nExtracted URL output:\n"
        f"{OUTPUT_URL_DIR}"
    )


if __name__ == "__main__":
    main()