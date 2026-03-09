"""Utility helpers for the electricity demand forecasting project."""

from pathlib import Path
import json


# Project root (one level up from src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def get_data_path() -> Path:
    """Find the dataset CSV, checking standard locations.

    Returns:
        Path to the dataset file.

    Raises:
        FileNotFoundError: If the dataset cannot be found.
    """
    candidates = [
        DATA_DIR / "continuous_dataset.csv",
        ROOT_DIR / "continuous_dataset.csv",
        ROOT_DIR / "continuous dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Dataset not found. Place 'continuous_dataset.csv' in the data/ directory. "
        f"Searched: {[str(p) for p in candidates]}"
    )


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load a JSON file as a dictionary."""
    with open(path, "r") as f:
        return json.load(f)
