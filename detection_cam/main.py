import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    raise SystemExit

paused = False
last_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire la frame.")
            break

        frame = cv2.flip(frame, 1)
        last_frame = frame.copy()
    else:
        frame = last_frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    status = "PAUSE" if paused else "RUNNING"

    cv2.putText(
        frame,
        f"Faces: {len(faces)} | {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Detection visage", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):  # BARRE ESPACE
        paused = not paused

    elif key == ord("q") or key == 27:  # Q ou ESC
        break

cap.release()
cv2.destroyAllWindows()