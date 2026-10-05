from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_EMAIL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "email"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "label_audit.csv"
)


def audit_dataset(file_path):

    df = pd.read_csv(
        file_path,
        low_memory=False,
        encoding_errors="replace"
    )

    dataset_name = file_path.stem.replace(
        "_normalized",
        ""
    )

    total = len(df)

    legitimate = (
        df["label"]
        .eq(0)
        .sum()
    )

    phishing = (
        df["label"]
        .eq(1)
        .sum()
    )

    unknown = (
        df["label"]
        .isna()
        .sum()
    )

    other = total - legitimate - phishing - unknown

    return {
        "dataset": dataset_name,
        "total_records": total,
        "legitimate": legitimate,
        "phishing_or_malicious": phishing,
        "unknown_labels": unknown,
        "other_labels": other,
        "phishing_percentage": (
            phishing / total * 100
            if total > 0
            else 0
        ),
    }


def main():

    print("=" * 80)
    print("LABEL AUDIT")
    print("=" * 80)

    files = sorted(
        PROCESSED_EMAIL_DIR.glob(
            "*_normalized.csv"
        )
    )

    results = []

    for file_path in files:

        try:

            result = audit_dataset(
                file_path
            )

            results.append(result)

            print(
                f"\n{result['dataset']}"
            )

            print(
                f"  Total      : "
                f"{result['total_records']:,}"
            )

            print(
                f"  Legitimate : "
                f"{result['legitimate']:,}"
            )

            print(
                f"  Phishing   : "
                f"{result['phishing_or_malicious']:,}"
            )

            print(
                f"  Unknown    : "
                f"{result['unknown_labels']:,}"
            )

            print(
                f"  Other      : "
                f"{result['other_labels']:,}"
            )

        except Exception as error:

            print(
                f"\nERROR: "
                f"{file_path.name}"
            )

            print(error)

    if results:

        report = pd.DataFrame(
            results
        )

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        report.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\n" + "=" * 80)
        print("LABEL AUDIT COMPLETE")
        print("=" * 80)

        print(
            f"\nReport saved to:\n"
            f"{OUTPUT_FILE}"
        )


if __name__ == "__main__":
    main()