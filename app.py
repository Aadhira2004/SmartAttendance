
import cv2
import numpy as np
import os
import csv
import threading

from flask import Flask, render_template, Response, jsonify
from datetime import datetime

app = Flask(__name__)

# ======================================
# PROJECT PATHS
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.npy")
CASCADE_PATH = os.path.join(
    BASE_DIR, "models", "haarcascade_frontalface_default.xml"
)

ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

# ======================================
# LOAD TRAINED MODEL
# ======================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

label_map = np.load(LABELS_PATH, allow_pickle=True).item()

reverse_labels = {
    value: key for key, value in label_map.items()
}

detector = cv2.CascadeClassifier(CASCADE_PATH)

if detector.empty():
    raise RuntimeError("Haar Cascade could not be loaded.")

# ======================================
# CAMERA SETTINGS
# ======================================

camera = None
camera_lock = threading.Lock()
camera_running = False

CONFIDENCE_THRESHOLD = 60

# ======================================
# ATTENDANCE FILE
# ======================================

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Person ID", "Role", "Date", "Time", "Status"
        ])


def mark_attendance(person_id):

    today = datetime.now().strftime("%Y-%m-%d")

    # Prevent duplicate attendance for the same person today
    with open(ATTENDANCE_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["Person ID"] == person_id
                and row["Date"] == today
            ):
                return False

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
            today,
            now.strftime("%H:%M:%S"),
            "Present"
        ])

    print("Attendance marked:", person_id)

    return True


# ======================================
# CAMERA FRAME GENERATOR
# ======================================

def generate_frames():

    global camera, camera_running

    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera = None
            camera_running = False
            return

    while camera_running:

        with camera_lock:
            if camera is None:
                break

            success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            if (
                confidence <= CONFIDENCE_THRESHOLD
                and label in reverse_labels
            ):

                person_id = reverse_labels[label]

                mark_attendance(person_id)

                text = f"{person_id} | {confidence:.1f}"
                color = (0, 220, 100)

            else:

                text = "Unknown"
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
                text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ======================================
# WEBSITE ROUTES
# ======================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start")
def start_camera():

    global camera_running

    camera_running = True

    return jsonify({"status": "Camera started"})


@app.route("/stop")
def stop_camera():

    global camera_running, camera

    camera_running = False

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

    return jsonify({"status": "Camera stopped"})


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ======================================
# ATTENDANCE API
# ======================================

@app.route("/api/attendance")
def get_attendance():

    records = []

    with open(ATTENDANCE_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            records.append(row)

    today = datetime.now().strftime("%Y-%m-%d")

    today_records = [
        row for row in records
        if row["Date"] == today
    ]

    return jsonify({
        "total": len(label_map),
        "present": len(today_records),
        "absent": max(0, len(label_map) - len(today_records)),
        "records": records[-50:][::-1],
        "people": list(label_map.keys())
    })


# ======================================
# RUN APPLICATION
# ======================================

if __name__ == "__main__":

    print("Smart Attendance Web Application")
    print("Registered people:", list(label_map.keys()))

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )

