from ultralytics import YOLO
import numpy as np

# Chargement du modèle une seule fois
model = YOLO("yolov8n.pt")


def detect_objects(image):
    """
    Prend une image PIL en entrée,
    retourne :
    - l'image annotée
    - la liste des détections
    - la liste des crops
    """
    img_array = np.array(image)

    results = model(img_array)
    result = results[0]

    annotated_image = result.plot()

    detections = []
    crops = []

    h, w = img_array.shape[:2]

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Sécurisation des coordonnées
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        detections.append({
            "id": i,
            "label": label,
            "confidence": round(conf, 3),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

        crop = img_array[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append({
                "id": i,
                "label": label,
                "confidence": round(conf, 3),
                "image": crop
            })

    return annotated_image, detections, crops