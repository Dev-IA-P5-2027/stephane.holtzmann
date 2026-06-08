import cv2

# Charge le modèle YuNet
model_path = "face_detection_yunet_2023mar.onnx"

# Ouvre la caméra 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la caméra 0.")
    exit()

# Lire une première frame pour récupérer la taille
ret, frame = cap.read()
if not ret:
    print("Erreur : impossible de lire la caméra.")
    cap.release()
    exit()

h, w = frame.shape[:2]

# Création du détecteur de visage
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(w, h),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)

print("Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture caméra.")
        break

    h, w = frame.shape[:2]

    # Important : mettre à jour la taille d'entrée si besoin
    detector.setInputSize((w, h))

    # Détection
    _, faces = detector.detect(frame)

    face_count = 0

    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            score = face[-1]

            face_count += 1

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Visage {score:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Nb visages: {face_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Detection de visages", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()