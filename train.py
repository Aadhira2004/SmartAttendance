

import cv2
import os
import numpy as np

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CASCADE_PATH = os.path.join(
    MODEL_DIR,
    "haarcascade_frontalface_default.xml"
)

TRAINED_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "trainer.yml"
)

LABELS_PATH = os.path.join(
    MODEL_DIR,
    "labels.npy"
)

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 2. LOAD HAAR CASCADE
# ==========================================

print("\nLoading Haar Cascade...")

detector = cv2.CascadeClassifier(CASCADE_PATH)

if detector.empty():
    raise FileNotFoundError(
        f"Haar Cascade file not found or invalid: {CASCADE_PATH}"
    )

print("Haar Cascade loaded successfully!")


# ==========================================
# 3. CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

print("\nReading dataset...")

faces = []
labels = []

label_map = {}
current_label = 0

valid_extensions = (".jpg", ".jpeg", ".png")


# ==========================================
# 4. READ IMAGES AND DETECT FACES
# ==========================================

for person_name in sorted(os.listdir(DATASET_DIR)):

    person_path = os.path.join(
        DATASET_DIR,
        person_name
    )

    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing: {person_name}")

    person_faces_found = 0

    for image_name in os.listdir(person_path):

        if not image_name.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(
            person_path,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"Cannot read image: {image_name}")
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(detected_faces) == 0:
            print(f"No face detected in: {image_name}")
            continue

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            # Resize every face to the same dimensions
            face = cv2.resize(
                face,
                (200, 200)
            )

            faces.append(face)

            if person_name not in label_map:
                label_map[person_name] = current_label
                current_label += 1

            labels.append(label_map[person_name])

            person_faces_found += 1

        print(
            f"Processed: {image_name} | "
            f"Faces detected: {len(detected_faces)}"
        )

    print(
        f"Total faces found for {person_name}: "
        f"{person_faces_found}"
    )


# ==========================================
# 5. VALIDATE TRAINING DATA
# ==========================================

if len(faces) == 0:
    raise ValueError(
        "No faces found. Check your dataset photographs."
    )

if len(label_map) < 2:
    raise ValueError(
        "At least TWO different people are required "
        "for LBPH training."
    )

faces = np.array(faces, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

print("\nDataset Summary")
print("------------------------------")
print("Total face samples:", len(faces))
print("Total people:", len(label_map))
print("People:", list(label_map.keys()))


# ==========================================
# 6. TRAIN LBPH FACE RECOGNIZER
# ==========================================

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "OpenCV Face module is missing. "
        "Install opencv-contrib-python."
    )

print("\nTraining LBPH Face Recognizer...")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(
    list(faces),
    labels
)

print("Training completed successfully!")


# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

recognizer.save(TRAINED_MODEL_PATH)

np.save(
    LABELS_PATH,
    label_map
)

print("\nModel saved successfully!")
print("------------------------------")
print("Trained Model:", TRAINED_MODEL_PATH)
print("Labels File:", LABELS_PATH)

print("\nPerson Labels:")

for name, label in label_map.items():
    print(f"{label} : {name}")


# ==========================================
# 8. FINAL STATUS
# ==========================================

print("\n================================")
print(" SMART ATTENDANCE TRAINING DONE")
print("================================")

