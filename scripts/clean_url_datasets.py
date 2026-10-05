from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import re
import pandas as pd


INPUT_DIR = Path("data/processed/url")
OUTPUT_DIR = Path("data/interim/url")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_url(url):
    """
    Create a conservative canonical representation of a URL.

    Important:
    We do NOT remove meaningful URL information such as:
    - path
    - query parameters
    - fragments

    because these can contain important phishing indicators.
    """

    if pd.isna(url):
        return ""

    url = str(url).strip()

    if not url:
        return ""

    # Remove surrounding whitespace
    url = url.strip()

    # Remove surrounding quotes if present
    url = url.strip("\"'")

    # Normalize whitespace
    url = re.sub(r"\s+", "", url)

    # Add a scheme temporarily if missing.
    # This allows urlsplit() to correctly identify the domain.
    parse_url = url

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", parse_url):
        parse_url = "http://" + parse_url

    try:
        parsed = urlsplit(parse_url)

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]

        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Remove trailing dot from hostname/domain
        if netloc.endswith("."):
            netloc = netloc[:-1]

        cleaned = urlunsplit(
            (
                scheme,
                netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )

        return cleaned

    except Exception:
        return url.lower()


def extract_domain(url):
    """Extract domain/netloc from a cleaned URL."""

    if not url:
        return ""

    try:
        parsed = urlsplit(url)

        domain = parsed.netloc.lower()

        # Remove credentials if present
        if "@" in domain:
            domain = domain.split("@")[-1]

        # Remove port
        domain = domain.split(":")[0]

        return domain

    except Exception:
        return ""


def process_dataset(input_file):

    print("\n" + "=" * 80)
    print(f"PROCESSING: {input_file.name}")
    print("=" * 80)

    df = pd.read_csv(input_file)

    original_count = len(df)

    # ---------------------------------------------------------
    # Canonical URL
    # ---------------------------------------------------------

    df["original_url"] = df["url"]

    df["clean_url"] = df["url"].apply(clean_url)

    # ---------------------------------------------------------
    # Domain
    # ---------------------------------------------------------

    df["domain"] = df["clean_url"].apply(extract_domain)

    # ---------------------------------------------------------
    # URL length
    # ---------------------------------------------------------

    df["url_length"] = df["clean_url"].str.len()

    # ---------------------------------------------------------
    # Domain length
    # ---------------------------------------------------------

    df["domain_length"] = df["domain"].str.len()

    # ---------------------------------------------------------
    # Remove rows where URL became empty
    # ---------------------------------------------------------

    df = df[
        df["clean_url"].notna()
        & (df["clean_url"].str.strip() != "")
    ].copy()

    # ---------------------------------------------------------
    # Remove exact duplicate records only
    # ---------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["source_dataset", "clean_url", "label"]
    ).copy()

    removed_duplicates = before_duplicates - len(df)

    # ---------------------------------------------------------
    # Final column order
    # ---------------------------------------------------------

    columns = [
        "sample_id",
        "source_dataset",
        "original_url",
        "clean_url",
        "domain",
        "url_length",
        "domain_length",
        "label",
    ]

    df = df[columns]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = OUTPUT_DIR / (
        input_file.stem.replace(
            "_normalized",
            "_cleaned"
        ) + ".csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Original rows        : {original_count:,}")
    print(f"Final rows           : {len(df):,}")
    print(f"Removed exact duplicates: {removed_duplicates:,}")
    print(f"Output               : {output_file}")

    print("\nLabel distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )


def main():

    print("=" * 80)
    print("URL CLEANING AND CANONICALIZATION")
    print("=" * 80)

    files = sorted(
        INPUT_DIR.glob("*_normalized.csv")
    )

    if not files:
        print("No normalized URL datasets found.")
        return

    for file_path in files:
        process_dataset(file_path)

    print("\n" + "=" * 80)
    print("URL CLEANING COMPLETE")
    print("=" * 80)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()