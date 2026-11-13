import cv2
import numpy as np
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

CASCADE = MODEL_DIR / "haarcascade_frontalface_default.xml"
MODEL = MODEL_DIR / "trainer.yml"
LABELS = MODEL_DIR / "labels.npy"

if not CASCADE.exists():
    raise SystemExit("ERROR: Haar cascade missing. Download it first.")

detector = cv2.CascadeClassifier(str(CASCADE))

if detector.empty():
    raise SystemExit("ERROR: Cannot load Haar detector.")

if not hasattr(cv2, "face"):
    raise SystemExit("ERROR: Install opencv-contrib-python.")

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

folders = sorted([p for p in DATASET.iterdir() if p.is_dir()])

for label, folder in enumerate(folders):
    label_map[label] = folder.name

    for image_path in folder.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read:", image_path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(50, 50)
        )

        if len(detected) == 0:
            print("No face found:", image_path)
            continue

        x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        faces.append(face)
        labels.append(label)

        print("Added:", folder.name, image_path.name)

if len(faces) < 2:
    raise SystemExit(
        "Not enough face images. Add more clear images per person."
    )

recognizer.train(faces, np.array(labels, dtype=np.int32))
recognizer.save(str(MODEL))
np.save(str(LABELS), label_map)

print("\nTRAINING COMPLETED")
print("Images:", len(faces))
print("People:", len(label_map))
print("Model:", MODEL)
print("Labels:", LABELS)
