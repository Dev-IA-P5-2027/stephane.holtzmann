from datetime import datetime
from pathlib import Path
from PIL import Image

from app.utils.paths import IMAGES_DIR
from app.repositories.image_repository import insert_image
from app.repositories.detection_repository import insert_detection
from app.services.detection_service import run_detection


def save_uploaded_image(uploaded_file) -> dict:
    """Sauvegarde une image uploadée sur disque."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    save_path = IMAGES_DIR / filename

    image = Image.open(uploaded_file).convert("RGB")
    image.save(save_path)

    return {
        "filename": filename,
        "original_path": str(save_path),
    }


def save_api_uploaded_file(uploaded_file) -> dict:
    """Sauvegarde un UploadFile FastAPI sur disque."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.filename}"
    save_path = IMAGES_DIR / filename

    with save_path.open("wb") as buffer:
        buffer.write(uploaded_file.file.read())

    return {
        "filename": filename,
        "original_path": str(save_path),
    }


def process_saved_image(saved: dict) -> dict:
    """Lance la détection sur une image déjà sauvegardée puis enregistre en base."""
    detection = run_detection(saved["original_path"])

    image_id = insert_image(
        filename=saved["filename"],
        original_path=saved["original_path"],
        model_name=detection["model_name"],
        confidence_threshold=0.0,
        total_detections=detection["total_detections"],
        annotated_path=detection["annotated_path"],
    )

    for det in detection["detections"]:
        insert_detection(
            image_id=image_id,
            label=det["label"],
            confidence=det["confidence"],
            x1=det["x1"],
            y1=det["y1"],
            x2=det["x2"],
            y2=det["y2"],
            crop_path=det["crop_path"],
        )

    return {
        "image_id": image_id,
        "filename": saved["filename"],
        "original_path": saved["original_path"],
        "annotated_path": detection["annotated_path"],
        "model_name": detection["model_name"],
        "total_detections": detection["total_detections"],
        "detections": detection["detections"],
    }


def process_uploaded_image(uploaded_file) -> dict:
    """Pipeline Streamlit."""
    saved = save_uploaded_image(uploaded_file)
    return process_saved_image(saved)


def process_uploaded_file_for_api(upload_file) -> dict:
    """Pipeline FastAPI."""
    saved = save_api_uploaded_file(upload_file)
    return process_saved_image(saved)