from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DIRECTORIES = [
    # Dataset layers
    "data/raw/email",
    "data/raw/url",
    "data/raw/threat_intelligence",
    "data/raw/benchmarks",

    "data/interim/email",
    "data/interim/url",
    "data/interim/extracted",

    "data/processed/email",
    "data/processed/url",
    "data/processed/master",

    "data/annotations",
    "data/splits",
    "data/metadata",

    # Research notebooks
    "notebooks",

    # Source code
    "src/phishing_detection/data",
    "src/phishing_detection/features",

    "src/phishing_detection/models/traditional_ml",
    "src/phishing_detection/models/transformer",
    "src/phishing_detection/models/url_detector",

    "src/phishing_detection/agents",

    "src/phishing_detection/explainability",

    "src/phishing_detection/orchestration",
    "src/phishing_detection/decision",
    "src/phishing_detection/evaluation",

    "src/phishing_detection/utils",

    # Experiments
    "experiments/baselines",
    "experiments/transformers",
    "experiments/agents",
    "experiments/adaptive",

    # Tests
    "tests",

    # Documentation
    "docs/research",
    "docs/architecture",
    "docs/experiments",

    # Scripts
    "scripts",

    # Frontend
    "frontend",
]


def create_directories():
    print("Creating project structure...\n")

    for directory in DIRECTORIES:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {directory}")

    print("\nProject structure created successfully.")


if __name__ == "__main__":
    create_directories()