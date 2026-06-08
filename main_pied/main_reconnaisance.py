import os
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
CAMERA_INDEX = 0
DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "known_faces"
COSINE_THRESHOLD = 0.363   # seuil de reconnaissance, à ajuster si besoin

# =========================
# VERIFICATIONS
# =========================
if not os.path.exists(DETECTOR_MODEL):
    print(f"Modèle introuvable : {DETECTOR_MODEL}")
    exit()

if not os.path.exists(RECOGNIZER_MODEL):
    print(f"Modèle introuvable : {RECOGNIZER_MODEL}")
    exit()

if not os.path.isdir(KNOWN_FACES_DIR):
    print(f"Dossier introuvable : {KNOWN_FACES_DIR}")
    exit()

# =========================
# OUVERTURE CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Erreur : impossible d'ouvrir la caméra {CAMERA_INDEX}.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Erreur : impossible de lire une image de la caméra.")
    cap.release()
    exit()

h, w = frame.shape[:2]

# =========================
# CHARGEMENT MODELES
# =========================
detector = cv2.FaceDetectorYN.create(
    model=DETECTOR_MODEL,
    config="",
    input_size=(w, h),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)

# =========================
# CHARGEMENT VISAGES CONNUS
# =========================
known_names = []
known_features = []

def extract_feature(image):
    """
    Détecte le premier visage de l'image et retourne son embedding.
    """
    ih, iw = image.shape[:2]
    detector.setInputSize((iw, ih))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # On prend le premier visage trouvé
    face = faces[0]
    aligned_face = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned_face)
    return feature

for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(KNOWN_FACES_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"Image illisible : {path}")
        continue

    feature = extract_feature(image)
    if feature is None:
        print(f"Aucun visage détecté dans : {filename}")
        continue

    name = os.path.splitext(filename)[0]
    known_names.append(name)
    known_features.append(feature)
    print(f"Visage chargé : {name}")

if len(known_features) == 0:
    print("Aucun visage connu valide trouvé dans le dossier known_faces.")
    cap.release()
    exit()

print("Appuie sur Q pour quitter.")

# =========================
# BOUCLE CAMERA
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture caméra.")
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)

            # Alignement + extraction caractéristiques
            aligned_face = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned_face)

            # Recherche du meilleur match
            best_score = -1.0
            best_name = "Inconnu"

            for name, known_feature in zip(known_names, known_features):
                score = recognizer.match(
                    feature,
                    known_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            # Décision
            if best_score >= COSINE_THRESHOLD:
                label = f"{best_name} ({best_score:.2f})"
            else:
                label = f"Inconnu ({best_score:.2f})"

            # Affichage
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Reconnaissance faciale", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()