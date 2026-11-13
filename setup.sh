
#!/bin/bash

set -e

echo "======================================"
echo " SMART ATTENDANCE PROJECT SETUP"
echo "======================================"

# 1. Create project
mkdir -p "$HOME/SmartAttendance"
cd "$HOME/SmartAttendance"

mkdir -p templates dataset models

# 2. Create Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install flask numpy opencv-contrib-python

# 4. Create backend
cat > app.py <<'PY'
from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import sqlite3
import base64
import os
import re
import csv
import io
from datetime import datetime, date

app = Flask(__name__)

DB = "attendance.db"
DATASET = "dataset"
MODEL = "models/face_model.yml"
LABELS = "models/labels.npy"
THRESHOLD = 65

os.makedirs(DATASET, exist_ok=True)
os.makedirs("models", exist_ok=True)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS students(
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            UNIQUE(student_id, date)
        )
    """)

    conn.commit()
    conn.close()

initialize_database()

def decode_image(data):
    if not isinstance(data, str) or not data:
        return None

    try:
        if "," in data:
            data = data.split(",", 1)[1]

        raw = base64.b64decode(data, validate=True)
        image = cv2.imdecode(
            np.frombuffer(raw, np.uint8),
            cv2.IMREAD_COLOR
        )
        return image
    except Exception:
        return None

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return gray, faces

def train_model():
    faces = []
    labels = []
    label_map = {}

    conn = get_db()
    students = conn.execute(
        "SELECT student_id FROM students ORDER BY student_id"
    ).fetchall()
    conn.close()

    for student in students:
        student_id = student["student_id"]
        folder = os.path.join(DATASET, student_id)

        if not os.path.isdir(folder):
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".jpg")
        ]

        # Require at least 3 samples per student
        if len(files) < 3:
            continue

        label = len(label_map)
        label_map[label] = student_id

        for filename in files:
            path = os.path.join(folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                faces.append(image)
                labels.append(label)

    if len(label_map) == 0:
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL)
    np.save(LABELS, label_map)

    return True

def load_model():
    if os.path.exists(MODEL) and os.path.exists(LABELS):
        recognizer.read(MODEL)
        return np.load(
            LABELS,
            allow_pickle=True
        ).item()
    return {}

label_map = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/register", methods=["POST"])
def register():
    global label_map

    data = request.get_json(silent=True) or {}

    student_id = data.get("student_id", "").strip()
    name = data.get("name", "").strip()
    image = decode_image(data.get("image", ""))

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", student_id):
        return jsonify({"message": "Invalid student ID."}), 400

    if not name or len(name) > 100:
        return jsonify({"message": "Enter a valid student name."}), 400

    if image is None:
        return jsonify({"message": "Invalid webcam image."}), 400

    gray, detected = detect_faces(image)

    if len(detected) != 1:
        return jsonify({
            "message": "Show exactly one face during registration."
        }), 400

    x, y, w, h = detected[0]

    face = cv2.resize(
        gray[y:y+h, x:x+w],
        (200, 200)
    )

    folder = os.path.join(DATASET, student_id)
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    ) + ".jpg"

    cv2.imwrite(
        os.path.join(folder, filename),
        face
    )

    conn = get_db()

    conn.execute("""
        INSERT INTO students(student_id, name)
        VALUES (?, ?)
        ON CONFLICT(student_id)
        DO UPDATE SET name=excluded.name
    """, (student_id, name))

    conn.commit()
    conn.close()

    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ])

    trained = train_model()

    if trained:
        label_map = load_model()

    return jsonify({
        "message": f"Sample saved for {name}. Total samples: {count}.",
        "samples": count,
        "trained": trained,
        "remaining": max(0, 3-count)
    })

@app.route("/api/scan", methods=["POST"])
def scan():
    global label_map

    if not os.path.exists(MODEL):
        return jsonify({
            "message": "Register at least 3 face samples per student first."
        }), 400

    data = request.get_json(silent=True) or {}
    image = decode_image(data.get("image", ""))

    if image is None:
        return jsonify({"message": "Invalid webcam image."}), 400

    gray, detected = detect_faces(image)

    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")

    results = []
    conn = get_db()

    for x, y, w, h in detected:
        face = cv2.resize(
            gray[y:y+h, x:x+w],
            (200, 200)
        )

        label, confidence = recognizer.predict(face)
        student_id = label_map.get(label)

        if student_id and confidence < THRESHOLD:

            student = conn.execute(
                "SELECT name FROM students WHERE student_id=?",
                (student_id,)
            ).fetchone()

            if student:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO attendance
                    (student_id, date, time, status)
                    VALUES (?, ?, ?, 'PRESENT')
                """, (student_id, today, now))

                results.append({
                    "student_id": student_id,
                    "name": student["name"],
                    "confidence": round(float(confidence), 2),
                    "status": "PRESENT",
                    "newly_marked": cursor.rowcount == 1
                })
                continue

        results.append({
            "student_id": "UNKNOWN",
            "name": "Unknown Face",
            "confidence": round(float(confidence), 2),
            "status": "NOT RECOGNIZED"
        })

    conn.commit()
    conn.close()

    return jsonify({
        "faces_detected": len(detected),
        "results": results,
        "message": "Scan completed."
    })

@app.route("/api/attendance")
def attendance():
    conn = get_db()

    rows = conn.execute("""
        SELECT s.student_id, s.name,
               a.date, a.time, a.status
        FROM attendance a
        JOIN students s
        ON a.student_id = s.student_id
        WHERE a.date=?
        ORDER BY a.time
    """, (date.today().isoformat(),)).fetchall()

    conn.close()

    return jsonify([dict(row) for row in rows])

@app.route("/api/stats")
def stats():
    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) FROM students"
    ).fetchone()[0]

    present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date=?",
        (date.today().isoformat(),)
    ).fetchone()[0]

    conn.close()

    return jsonify({
        "total_students": total,
        "present": present,
        "absent": max(0, total-present)
    })

@app.route("/api/export")
def export():
    conn = get_db()

    rows = conn.execute("""
        SELECT s.student_id, s.name,
               a.date, a.time, a.status
        FROM attendance a
        JOIN students s
        ON a.student_id=s.student_id
        ORDER BY a.date DESC
    """).fetchall()

    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Student ID", "Name", "Date", "Time", "Status"
    ])

    for row in rows:
        writer.writerow(list(row))

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=attendance_report.csv"
        }
    )

if __name__ == "__main__":
    print("SMART ATTENDANCE SERVER STARTING...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
PY

# 5. Create complete frontend
cat > templates/index.html <<'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Attendance System</title>

<style>
* { box-sizing:border-box; }

body {
    margin:0;
    font-family:Arial,sans-serif;
    background:#f3f6fb;
    color:#1c2d45;
}

header {
    background:#142b4a;
    color:white;
    padding:25px 6%;
}

header h1 { margin:0; font-size:25px; }
header p { color:#ccd9e9; }

.container {
    max-width:1150px;
    margin:30px auto;
    padding:15px;
}

.stats {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:15px;
    margin-bottom:20px;
}

.stat,.card {
    background:white;
    padding:22px;
    border-radius:12px;
    box-shadow:0 3px 15px #0000000a;
}

.stat h2 {
    font-size:30px;
    margin:8px 0;
    color:#176b58;
}

.grid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:20px;
}

video {
    width:100%;
    border-radius:10px;
    background:#111;
}

input {
    display:block;
    width:100%;
    padding:13px;
    margin:12px 0;
    border:1px solid #d5dce6;
    border-radius:7px;
}

button {
    background:#176b58;
    color:white;
    border:0;
    border-radius:7px;
    padding:12px 17px;
    margin:5px 3px 5px 0;
    cursor:pointer;
    font-weight:bold;
}

button:hover { opacity:.85; }
.stop { background:#bd4545; }
.blue { background:#285d9c; }

.message {
    background:#eef6f2;
    padding:12px;
    border-radius:7px;
    margin-top:12px;
    font-size:14px;
}

table {
    width:100%;
    border-collapse:collapse;
    margin-top:18px;
}

th,td {
    padding:13px;
    border-bottom:1px solid #e6ebf1;
    text-align:left;
}

th { background:#edf2f8; }

.table-wrap { overflow-x:auto; }

@media(max-width:750px) {
    .grid { grid-template-columns:1fr; }
    .stats { grid-template-columns:1fr; }
}
</style>
</head>

<body>

<header>
<h1>SMART ATTENDANCE SYSTEM</h1>
<p>Automated Face Recognition Attendance Management</p>
</header>

<main class="container">

<section class="stats">
<div class="stat">
Total Students
<h2 id="total">0</h2>
</div>
<div class="stat">
Present Today
<h2 id="present">0</h2>
</div>
<div class="stat">
Absent Today
<h2 id="absent">0</h2>
</div>
</section>

<section class="grid">

<div class="card">
<h3>Live Classroom Camera</h3>

<video id="video" autoplay playsinline muted></video>
<canvas id="canvas" hidden></canvas>

<button onclick="startCamera()">Start Camera</button>
<button class="stop" onclick="stopCamera()">Stop Camera</button>
<button class="blue" onclick="scanClass()">Scan Classroom</button>

<div class="message" id="message">
Camera not started.
</div>
</div>

<div class="card">
<h3>Register Student</h3>

<input id="student_id" placeholder="Student ID (STU001)">
<input id="name" placeholder="Student Name">

<p>Only one face should be visible during registration.</p>

<button onclick="registerStudent()">
Capture Face Sample
</button>

<div class="message" id="registerMessage">
Capture at least 3 samples for each student.
</div>
</div>

</section>

<section class="card" style="margin-top:22px">

<h3>Today's Attendance Records</h3>

<button class="blue" onclick="loadAttendance()">
Refresh Attendance
</button>

<a href="/api/export">
<button>Download CSV Report</button>
</a>

<div class="table-wrap">
<table>
<thead>
<tr>
<th>Student ID</th>
<th>Name</th>
<th>Date</th>
<th>Time</th>
<th>Status</th>
</tr>
</thead>
<tbody id="attendanceBody"></tbody>
</table>
</div>

</section>
</main>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");

let cameraStream = null;
let busy = false;

async function startCamera() {
    try {
        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {
            throw new Error("Webcam unavailable. Use Chrome or Safari on localhost.");
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {facingMode:"user"},
            audio:false
        });

        video.srcObject = cameraStream;
        await video.play();

        document.getElementById("message").innerText =
            "Camera started successfully.";

    } catch(error) {
        console.error("Camera Error:",error);
        document.getElementById("message").innerText =
            error.name + ": " + error.message;
    }
}

function stopCamera() {
    if(cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        video.srcObject = null;
        document.getElementById("message").innerText =
            "Camera stopped.";
    }
}

function captureImage() {
    if(!cameraStream || video.videoWidth === 0) {
        throw new Error("Please start the camera first.");
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video,0,0);

    return canvas.toDataURL("image/jpeg",0.85);
}

async function registerStudent() {
    try {
        const student_id =
            document.getElementById("student_id").value.trim();

        const name =
            document.getElementById("name").value.trim();

        if(!student_id || !name) {
            throw new Error("Enter student ID and name.");
        }

        const response = await fetch("/api/register",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({
                student_id,
                name,
                image:captureImage()
            })
        });

        const result = await response.json();

        if(!response.ok) throw new Error(result.message);

        document.getElementById("registerMessage").innerText =
            result.message +
            (result.trained
                ? " Recognition model updated."
                : " More samples needed before training.");

    } catch(error) {
        document.getElementById("registerMessage").innerText =
            error.message;
    }
}

async function scanClass() {
    if(busy) return;

    try {
        busy = true;

        document.getElementById("message").innerText =
            "Scanning faces...";

        const response = await fetch("/api/scan",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({
                image:captureImage()
            })
        });

        const result = await response.json();

        if(!response.ok) throw new Error(result.message);

        const recognized = result.results.map(item =>
            item.name + ": " + item.status
        ).join(" | ");

        document.getElementById("message").innerText =
            "Faces detected: " + result.faces_detected +
            (recognized ? " | " + recognized : "");

        await loadAttendance();
        await loadStats();

    } catch(error) {
        document.getElementById("message").innerText =
            error.message;
    } finally {
        busy = false;
    }
}

async function loadAttendance() {
    try {
        const response = await fetch("/api/attendance");
        const records = await response.json();

        const tbody =
            document.getElementById("attendanceBody");

        tbody.innerHTML = "";

        records.forEach(record => {
            const row = document.createElement("tr");

            [
                record.student_id,
                record.name,
                record.date,
                record.time,
                record.status
            ].forEach(value => {
                const cell = document.createElement("td");
                cell.textContent = value;
                row.appendChild(cell);
            });

            tbody.appendChild(row);
        });
    } catch(error) {
        console.error(error);
    }
}

async function loadStats() {
    try {
        const response = await fetch("/api/stats");
        const data = await response.json();

        document.getElementById("total").innerText =
            data.total_students;

        document.getElementById("present").innerText =
            data.present;

        document.getElementById("absent").innerText =
            data.absent;
    } catch(error) {
        console.error(error);
    }
}

loadAttendance();
loadStats();

setInterval(() => {
    loadAttendance();
    loadStats();
},10000);

</script>
</body>
</html>
HTML

# 6. Create run script
cat > run.sh <<'SH'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
SH

chmod +x run.sh

# 7. Verify OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__); print('Face module:', hasattr(cv2, 'face'))"

echo ""
echo "======================================"
echo " PROJECT CREATED SUCCESSFULLY!"
echo "======================================"
echo "Starting website..."
echo "Open: http://127.0.0.1:5000"
echo ""

python app.py

