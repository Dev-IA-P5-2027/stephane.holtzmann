import cv2

# Ouvre la caméra par défaut (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la caméra.")
    exit()

# Détecteur de personnes basé sur HOG + SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire l'image de la caméra.")
        break

    # Redimensionne pour accélérer un peu
    frame = cv2.resize(frame, (800, 600))

    # Détection des personnes
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # Dessine les rectangles
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Personne", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Affiche le nombre de personnes détectées
    cv2.putText(frame, f"Nb personnes: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detection de personnes", frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()