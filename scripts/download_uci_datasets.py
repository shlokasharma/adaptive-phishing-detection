from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "url"
)

URL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_dataset(dataset_id, dataset_name):

    print("\n" + "=" * 80)
    print(f"DOWNLOADING: {dataset_name}")
    print(f"UCI DATASET ID: {dataset_id}")
    print("=" * 80)

    try:

        dataset = fetch_ucirepo(
            id=dataset_id
        )

        features = dataset.data.features
        targets = dataset.data.targets

        print(
            f"\nFeatures shape: "
            f"{features.shape}"
        )

        print(
            f"Target shape: "
            f"{targets.shape}"
        )

        # Combine features and target
        df = pd.concat(
            [
                features.reset_index(drop=True),
                targets.reset_index(drop=True)
            ],
            axis=1
        )

        output_file = (
            URL_DIR
            / f"{dataset_name}.csv"
        )

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        print(
            f"\nSaved successfully:"
        )

        print(output_file)

        print(
            f"\nFinal shape: "
            f"{df.shape}"
        )

        return True

    except Exception as error:

        print(
            f"\nERROR downloading "
            f"{dataset_name}"
        )

        print(error)

        return False


def main():

    print("=" * 80)
    print("UCI PHISHING DATASET ACQUISITION")
    print("=" * 80)

    results = []

    # UCI Phishing Websites
    results.append(
        save_dataset(
            327,
            "uci_phishing_websites"
        )
    )

    # UCI PhiUSIIL
    results.append(
        save_dataset(
            967,
            "phiusiil"
        )
    )

    print("\n" + "=" * 80)
    print("UCI DATASET ACQUISITION COMPLETE")
    print("=" * 80)

    print(
        f"\nSuccessful datasets: "
        f"{sum(results)}/{len(results)}"
    )


if __name__ == "__main__":
    main()