
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.npy")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "haarcascade_frontalface_default.xml"
)

ATTENDANCE_FILE = os.path.join(
    BASE_DIR,
    "attendance.csv"
)

# =====================================
# 2. CHECK REQUIRED FILES
# =====================================

for path in [MODEL_PATH, LABELS_PATH, CASCADE_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

label_map = np.load(
    LABELS_PATH,
    allow_pickle=True
).item()

reverse_labels = {
    value: key for key, value in label_map.items()
}

detector = cv2.CascadeClassifier(CASCADE_PATH)

if detector.empty():
    raise RuntimeError("Could not load Haar Cascade.")

print("Face recognition model loaded!")
print("Registered people:", list(label_map.keys()))

# =====================================
# 4. CREATE ATTENDANCE FILE
# =====================================

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Person ID",
            "Role",
            "Date",
            "Time",
            "Status"
        ])

# Avoid duplicate entries during this session
marked_people = set()

# =====================================
# 5. ATTENDANCE FUNCTION
# =====================================

def mark_attendance(person_id):

    if person_id in marked_people:
        return

    role = (
        "FACULTY"
        if person_id.startswith("FACULTY")
        else "STUDENT"
    )

    now = datetime.now()

    with open(ATTENDANCE_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            person_id,
            role,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "Present"
        ])

    marked_people.add(person_id)

    print(f"Attendance marked: {person_id} ({role})")

# =====================================
# 6. OPEN WEBCAM
# =====================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot access webcam. Check camera permissions."
    )

print("\nWebcam started!")
print("Press Q to quit.")

# =====================================
# 7. LIVE FACE RECOGNITION
# =====================================

CONFIDENCE_THRESHOLD = 60

try:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Cannot read webcam frame.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (200, 200)
            )

            label, confidence = recognizer.predict(face)

            if (
                confidence <= CONFIDENCE_THRESHOLD
                and label in reverse_labels
            ):

                person_id = reverse_labels[label]

                mark_attendance(person_id)

                display_text = (
                    f"{person_id} | {confidence:.1f}"
                )

                color = (0, 255, 0)

            else:

                display_text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                display_text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow(
            "Smart Attendance System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("\nWebcam stopped.")
    print("Attendance saved in attendance.csv")

