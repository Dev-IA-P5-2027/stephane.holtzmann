import os
import time
from collections import deque

import cv2
import numpy as np

# =========================
# IMPORTS OPTIONNELS
# =========================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# =========================
# CONFIG GENERALE
# =========================
CAMERA_INDEX = 0
WINDOW_NAME = "Detection Benchmark Lab"

RESOLUTIONS = [
    (640, 480),
    (960, 540),
    (1280, 720),
]

INITIAL_RESOLUTION_INDEX = 0
INITIAL_FLIP_IMAGE = True

# Fréquence de calcul des modules
FACE_EVERY_N_FRAMES = 1
HAND_EVERY_N_FRAMES = 1
POSE_EVERY_N_FRAMES = 1

# Affichage
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.55
TEXT_THICKNESS = 1
FPS_HISTORY_SIZE = 30

# =========================
# CONFIG FACE RECO
# =========================
DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "known_faces"

# Plus haut = plus strict
COSINE_THRESHOLD = 0.45


# =========================
# OUTILS GENERAUX
# =========================
def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def draw_label(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(img, text, (x, y), FONT, TEXT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA)


def add_header(img, title, color=(255, 255, 255)):
    header = img.copy()
    cv2.rectangle(header, (0, 0), (img.shape[1], 28), (30, 30, 30), -1)
    draw_label(header, title, 10, 20, color)
    return header


def pad_to_same_size(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def build_mosaic(images, cell_size=(640, 360)):
    target_w, target_h = cell_size
    processed = [pad_to_same_size(img, target_w, target_h) for img in images]

    top = np.hstack((processed[0], processed[1]))
    bottom = np.hstack((processed[2], processed[3]))
    return np.vstack((top, bottom))


def get_cpu_usage():
    if PSUTIL_AVAILABLE:
        return psutil.cpu_percent(interval=None)
    return None


# =========================
# FACE RECO UTILS
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


def load_known_faces(detector, recognizer):
    known_faces = []
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"[WARN] Dossier introuvable : {KNOWN_FACES_DIR}")
        return known_faces

    files = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith(valid_exts)
    ]

    if not files:
        print("[WARN] Aucune image trouvée dans known_faces.")
        return known_faces

    for filename in files:
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"[WARN] Image illisible : {filename}")
            continue

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is None or len(faces) == 0:
            print(f"[WARN] Aucun visage détecté dans : {filename}")
            continue

        face = faces[0]

        try:
            aligned_face = recognizer.alignCrop(img, face)
            embedding = recognizer.feature(aligned_face)
        except Exception as e:
            print(f"[WARN] Erreur traitement {filename} : {e}")
            continue

        name = os.path.splitext(filename)[0]
        known_faces.append({
            "name": name,
            "embedding": embedding
        })

        print(f"[OK] Visage connu chargé : {name}")

    return known_faces


# =========================
# INITIALISATION FACE RECO
# =========================
if os.path.exists(DETECTOR_MODEL) and os.path.exists(RECOGNIZER_MODEL):
    try:
        yunet_detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            (320, 320),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000
        )

        face_recognizer = cv2.FaceRecognizerSF.create(
            RECOGNIZER_MODEL,
            ""
        )

        known_faces = load_known_faces(yunet_detector, face_recognizer)
    except Exception as e:
        print(f"[WARN] Initialisation Face Reco impossible : {e}")
        yunet_detector = None
        face_recognizer = None
        known_faces = []
else:
    yunet_detector = None
    face_recognizer = None
    known_faces = []
    print("[WARN] Modèles visage/reco introuvables. Module reconnaissance faciale désactivé.")


# =========================
# INITIALISATION MEDIAPIPE
# =========================
if MEDIAPIPE_AVAILABLE:
    try:
        # Compatibilité ancienne API
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        print(f"[WARN] Mediapipe disponible mais API incompatible : {e}")
        MEDIAPIPE_AVAILABLE = False
        hands = None
        pose = None
        mp_hands = None
        mp_pose = None
        mp_drawing = None
else:
    hands = None
    pose = None
    mp_hands = None
    mp_pose = None
    mp_drawing = None


# =========================
# PIPELINES
# =========================
def pipeline_original(frame):
    out = frame.copy()
    out = add_header(out, "Vue originale")
    return out, 0.0

def draw_face_emoji(img, x, y, w, h, label="🙂"):
    """
    Dessine un emoji/visage cartoon par-dessus la zone du visage.
    Basé sur la bbox détectée.
    """
    # Centre du visage
    cx = x + w // 2
    cy = y + h // 2

    # Rayon principal
    radius = max(10, min(w, h) // 2)

    # Fond du visage (jaune)
    cv2.circle(img, (cx, cy), radius, (0, 255, 255), -1)
    cv2.circle(img, (cx, cy), radius, (0, 0, 0), 2)

    # Yeux
    eye_y = cy - h // 6
    eye_dx = w // 5
    eye_r = max(2, radius // 10)

    left_eye = (cx - eye_dx, eye_y)
    right_eye = (cx + eye_dx, eye_y)

    cv2.circle(img, left_eye, eye_r, (0, 0, 0), -1)
    cv2.circle(img, right_eye, eye_r, (0, 0, 0), -1)

    # Bouche (sourire)
    mouth_y = cy + h // 8
    mouth_w = max(10, w // 3)
    mouth_h = max(6, h // 6)

    cv2.ellipse(
        img,
        (cx, mouth_y),
        (mouth_w // 2, mouth_h // 2),
        0,
        0,
        180,
        (0, 0, 0),
        2
    )

    # Petit texte du label au-dessus
    cv2.putText(
        img,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA)

def pipeline_face_recognition(frame):
    start = time.perf_counter()
    out = frame.copy()

    if yunet_detector is None or face_recognizer is None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        out = add_header(out, f"Face Reco OFF | {elapsed_ms:.1f} ms | modeles absents")
        return out, elapsed_ms

    h, w = out.shape[:2]
    yunet_detector.setInputSize((w, h))
    _, faces = yunet_detector.detect(out)

    face_count = 0
    recognized_count = 0

    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            face_count += 1

            try:
                aligned_face = face_recognizer.alignCrop(out, face)
                embedding = face_recognizer.feature(aligned_face)
                label, sim_score = match_face(
                    embedding,
                    known_faces,
                    threshold=COSINE_THRESHOLD
                )
            except Exception:
                label = "Erreur"
                sim_score = 0.0

            if label not in ["Inconnu", "Erreur"]:
                recognized_count += 1

            # On masque/replace le visage UNIQUEMENT dans cette vue
            # en dessinant un emoji qui suit la bbox.
            if label == "Steph":
                emoji_text = "Steph 😄"
            elif label == "jimmy":
                emoji_text = "Jimmy 😎"
            elif label == "Inconnu":
                emoji_text = "Inconnu 🙂"
            else:
                emoji_text = "Erreur 😵"

            draw_face_emoji(out, x, y, fw, fh, emoji_text)

            # Optionnel : afficher le score en dessous
            cv2.putText(
                out,
                f"sim={sim_score:.2f}",
                (x, min(h - 10, y + fh + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    elapsed_ms = (time.perf_counter() - start) * 1000
    title = f"Face Reco | {elapsed_ms:.1f} ms | nb={face_count} | reco={recognized_count}"
    out = add_header(out, title)
    return out, elapsed_ms


def pipeline_hands(frame):
    start = time.perf_counter()
    out = frame.copy()
    detected_hands = 0

    if MEDIAPIPE_AVAILABLE and hands is not None:
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            detected_hands = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    out,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

    elapsed_ms = (time.perf_counter() - start) * 1000
    title = f"Mains | {elapsed_ms:.1f} ms | nb={detected_hands}"
    if not MEDIAPIPE_AVAILABLE:
        title += " | mediapipe off"

    out = add_header(out, title)
    return out, elapsed_ms


def pipeline_pose(frame):
    start = time.perf_counter()
    out = frame.copy()
    pose_found = False

    if MEDIAPIPE_AVAILABLE and pose is not None:
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            pose_found = True
            mp_drawing.draw_landmarks(
                out,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

    elapsed_ms = (time.perf_counter() - start) * 1000
    title = f"Pose | {elapsed_ms:.1f} ms | ok={pose_found}"
    if not MEDIAPIPE_AVAILABLE:
        title += " | mediapipe off"

    out = add_header(out, title)
    return out, elapsed_ms


# =========================
# MAIN
# =========================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    resolution_index = INITIAL_RESOLUTION_INDEX
    width, height = RESOLUTIONS[resolution_index]
    set_resolution(cap, width, height)

    flip_image = INITIAL_FLIP_IMAGE

    face_enabled = True
    hands_enabled = True
    pose_enabled = True
    show_original = True

    last_face_view = None
    last_hands_view = None
    last_pose_view = None

    last_face_ms = 0.0
    last_hands_ms = 0.0
    last_pose_ms = 0.0

    frame_count = 0
    fps_history = deque(maxlen=FPS_HISTORY_SIZE)
    prev_time = time.perf_counter()

    print("Commandes clavier :")
    print("  q  -> quitter")
    print("  1  -> activer/desactiver face recognition")
    print("  2  -> activer/desactiver mains")
    print("  3  -> activer/desactiver pose")
    print("  o  -> activer/desactiver vue originale")
    print("  r  -> changer resolution")
    print("  f  -> activer/desactiver flip horizontal")
    print()

    while True:
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire la frame.")
            break

        frame_count += 1

        if flip_image:
            frame = cv2.flip(frame, 1)

        # Vue originale
        if show_original:
            original_view, _ = pipeline_original(frame)
        else:
            original_view = np.zeros_like(frame)
            original_view = add_header(original_view, "Vue originale OFF")

        # Face recognition
        if face_enabled:
            if frame_count % FACE_EVERY_N_FRAMES == 0 or last_face_view is None:
                last_face_view, last_face_ms = pipeline_face_recognition(frame)
            face_view = last_face_view.copy()
        else:
            face_view = np.zeros_like(frame)
            face_view = add_header(face_view, "Face Reco OFF")
            last_face_ms = 0.0

        # Hands
        if hands_enabled:
            if frame_count % HAND_EVERY_N_FRAMES == 0 or last_hands_view is None:
                last_hands_view, last_hands_ms = pipeline_hands(frame)
            hands_view = last_hands_view.copy()
        else:
            hands_view = np.zeros_like(frame)
            hands_view = add_header(hands_view, "Mains OFF")
            last_hands_ms = 0.0

        # Pose
        if pose_enabled:
            if frame_count % POSE_EVERY_N_FRAMES == 0 or last_pose_view is None:
                last_pose_view, last_pose_ms = pipeline_pose(frame)
            pose_view = last_pose_view.copy()
        else:
            pose_view = np.zeros_like(frame)
            pose_view = add_header(pose_view, "Pose OFF")
            last_pose_ms = 0.0

        # Mosaïque
        mosaic = build_mosaic(
            [original_view, face_view, hands_view, pose_view],
            cell_size=(640, 360)
        )

        # FPS
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        current_fps = 1.0 / dt if dt > 0 else 0.0
        fps_history.append(current_fps)
        avg_fps = sum(fps_history) / len(fps_history)

        # Temps de boucle
        loop_ms = (time.perf_counter() - loop_start) * 1000

        # CPU
        cpu_usage = get_cpu_usage()

        # Overlay global
        overlay_y = 30
        draw_label(mosaic, f"FPS moyen: {avg_fps:.1f}", 10, overlay_y, (0, 255, 255))
        overlay_y += 22
        draw_label(mosaic, f"Loop: {loop_ms:.1f} ms", 10, overlay_y, (0, 255, 255))
        overlay_y += 22
        draw_label(mosaic, f"Face Reco: {last_face_ms:.1f} ms", 10, overlay_y, (0, 255, 255))
        overlay_y += 22
        draw_label(mosaic, f"Hands: {last_hands_ms:.1f} ms", 10, overlay_y, (0, 255, 255))
        overlay_y += 22
        draw_label(mosaic, f"Pose: {last_pose_ms:.1f} ms", 10, overlay_y, (0, 255, 255))
        overlay_y += 22

        if cpu_usage is not None:
            draw_label(mosaic, f"CPU: {cpu_usage:.1f} %", 10, overlay_y, (0, 255, 255))
            overlay_y += 22

        draw_label(
            mosaic,
            f"Resolution: {width}x{height} | Face[{face_enabled}] Hands[{hands_enabled}] Pose[{pose_enabled}]",
            10,
            overlay_y,
            (0, 255, 255)
        )

        cv2.imshow(WINDOW_NAME, mosaic)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("1"):
            face_enabled = not face_enabled
        elif key == ord("2"):
            hands_enabled = not hands_enabled
        elif key == ord("3"):
            pose_enabled = not pose_enabled
        elif key == ord("o"):
            show_original = not show_original
        elif key == ord("r"):
            resolution_index = (resolution_index + 1) % len(RESOLUTIONS)
            width, height = RESOLUTIONS[resolution_index]
            set_resolution(cap, width, height)
            print(f"Resolution changee : {width}x{height}")
        elif key == ord("f"):
            flip_image = not flip_image

    cap.release()
    cv2.destroyAllWindows()

    if MEDIAPIPE_AVAILABLE:
        try:
            if hands is not None:
                hands.close()
            if pose is not None:
                pose.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()