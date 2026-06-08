from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

IMAGES_DIR = DATA_DIR / "images"
ANNOTATED_DIR = DATA_DIR / "annotated"
CROPS_DIR = DATA_DIR / "crops"
EXPORTS_DIR = DATA_DIR / "exports"

# Dataset structuré (anticipation training)
DATASET_DIR = DATA_DIR / "dataset"
DATASET_RAW_DIR = DATASET_DIR / "raw"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
DATASET_LABELS_DIR = DATASET_DIR / "labels"


def ensure_directories() -> None:
    """Crée les dossiers nécessaires si absents."""
    directories = [
        DATA_DIR,
        IMAGES_DIR,
        ANNOTATED_DIR,
        CROPS_DIR,
        EXPORTS_DIR,
        DATASET_DIR,
        DATASET_RAW_DIR,
        DATASET_IMAGES_DIR,
        DATASET_LABELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)