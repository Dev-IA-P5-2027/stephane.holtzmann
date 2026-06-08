import cv2
from ultralytics import YOLO

# Charge le dernier modèle YOLO officiel d'Ultralytics
# Version nano = plus légère et plus rapide
model = YOLO("yolo26n.pt")

# Ouvre la caméra index 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la caméra 0.")
    exit()

print("Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire l'image de la caméra.")
        break

    # Prédiction sur l'image
    results = model(frame, verbose=False)

    person_count = 0

    # Parcours des détections
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # COCO: classe 0 = person
            if cls == 0 and conf > 0.5:
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Personne {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Affichage du compteur
    cv2.putText(
        frame,
        f"Nb personnes: {person_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Detection de personnes - YOLO26", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()