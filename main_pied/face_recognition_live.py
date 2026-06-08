import os
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "known_faces"
CAMERA_INDEX = 0

# Plus haut = plus strict
COSINE_THRESHOLD = 0.45


# =========================
# VERIFICATIONS
# =========================
if not os.path.exists(DETECTOR_MODEL):
    print(f"Erreur : modèle détecteur introuvable : {DETECTOR_MODEL}")
    raise SystemExit

if not os.path.exists(RECOGNIZER_MODEL):
    print(f"Erreur : modèle reconnaissance introuvable : {RECOGNIZER_MODEL}")
    raise SystemExit

if not os.path.isdir(KNOWN_FACES_DIR):
    print(f"Erreur : dossier introuvable : {KNOWN_FACES_DIR}")
    raise SystemExit


# =========================
# OUTILS
# =========================
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32).flatten()
    v2 = np.array(vec2, dtype=np.float32).flatten()

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)


def match_face(embedding, known_faces, threshold=0.45):
    best_name = "Inconnu"
    best_score = -1.0

    for item in known_faces:
        score = cosine_similarity(embedding, item["embedding"])

        if score > best_score:
            best_score = score
            best_name = item["name"]

    if best_score >= threshold:
        return best_name, best_score

    return "Inconnu", best_score


# =========================
# OUVERTURE CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Erreur : impossible d'ouvrir la caméra {CAMERA_INDEX}.")
    raise SystemExit

# Lire une première frame pour récupérer la taille
ret, frame = cap.read()
if not ret:
    print("Erreur : impossible de lire la caméra.")
    cap.release()
    raise SystemExit

h, w = frame.shape[:2]


# =========================
# CREATION DES MODELES
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
# CHARGEMENT DES VISAGES CONNUS
# =========================
known_faces = []
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

files = [
    f for f in os.listdir(KNOWN_FACES_DIR)
    if f.lower().endswith(valid_exts)
]

if not files:
    print("Erreur : aucune image trouvée dans known_faces.")
    cap.release()
    raise SystemExit

for filename in files:
    path = os.path.join(KNOWN_FACES_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"[IGNORÉ] Image illisible : {filename}")
        continue

    ih, iw = img.shape[:2]
    detector.setInputSize((iw, ih))
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        print(f"[IGNORÉ] Aucun visage détecté dans : {filename}")
        continue

    # On prend le premier visage trouvé
    face = faces[0]

    try:
        aligned_face = recognizer.alignCrop(img, face)
        embedding = recognizer.feature(aligned_face)
    except Exception as e:
        print(f"[IGNORÉ] Erreur traitement {filename} : {e}")
        continue

    name = os.path.splitext(filename)[0]

    known_faces.append({
        "name": name,
        "embedding": embedding
    })

    print(f"[OK] Visage connu chargé : {name}")

if not known_faces:
    print("Erreur : aucun visage exploitable chargé depuis known_faces.")
    cap.release()
    raise SystemExit


# =========================
# BOUCLE CAMERA
# =========================
print("Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture caméra.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Mettre à jour la taille d'entrée
    detector.setInputSize((w, h))

    # Détection
    _, faces = detector.detect(frame)

    face_count = 0

    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            detect_score = float(face[-1])

            face_count += 1

            try:
                aligned_face = recognizer.alignCrop(frame, face)
                embedding = recognizer.feature(aligned_face)

                label, sim_score = match_face(
                    embedding,
                    known_faces,
                    threshold=COSINE_THRESHOLD
                )
            except Exception:
                label = "Erreur"
                sim_score = 0.0

            # Couleur : vert si reconnu, rouge sinon
            color = (0, 255, 0) if label not in ["Inconnu", "Erreur"] else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            cv2.putText(
                frame,
                f"{label} | det={detect_score:.2f} | sim={sim_score:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    cv2.putText(
        frame,
        f"Nb visages: {face_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Threshold reco: {COSINE_THRESHOLD:.2f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Detection + Reconnaissance faciale", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()