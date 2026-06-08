from pathlib import Path

import cv2
from ultralytics import YOLO

from app.utils.paths import ANNOTATED_DIR, CROPS_DIR

model = YOLO("yolov8n.pt")


def run_detection(image_path: str) -> dict:
    """
    Effectue la détection sur une image déjà enregistrée sur disque.
    Sauvegarde l'image annotée et les crops.
    Retourne les métadonnées de détection.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    results = model(str(image_path))
    result = results[0]

    annotated = result.plot()
    annotated_path = ANNOTATED_DIR / image_path.name
    cv2.imwrite(str(annotated_path), annotated)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    detections = []

    crop_dir = CROPS_DIR / image_path.stem
    crop_dir.mkdir(parents=True, exist_ok=True)

    if result.boxes is not None:
        for i, (box, cls, conf) in enumerate(
            zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf),
            start=1
        ):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(cls.item())
            label = model.names[class_id]
            confidence = float(conf.item())

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))

            crop_path = None

            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crop_filename = f"{i}_{label}.jpg"
                crop_full_path = crop_dir / crop_filename
                cv2.imwrite(str(crop_full_path), crop)
                crop_path = str(crop_full_path)

            detections.append({
                "label": label,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": crop_path,
            })

    return {
        "model_name": "yolov8n",
        "total_detections": len(detections),
        "annotated_path": str(annotated_path),
        "detections": detections,
    }