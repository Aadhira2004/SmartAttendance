import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import threading
import av
import os
from pathlib import Path
from datetime import datetime, date
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📷",
    layout="wide"
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "attendance_streamlit.db"

# ==========================================
# GLOBAL RECOGNITION STATE
# ==========================================

STATE_LOCK = threading.Lock()

STATE = {
    "name": "Waiting for camera",
    "confidence": None,
    "status": "Waiting",
    "last_seen": None,
    "message": "Start the camera to begin"
}

RECOGNIZER = None
FACE_CASCADE = None
LABEL_MAP = {}
MODEL_ERROR = None
CONFIDENCE_LIMIT = 70.0

# ==========================================
# DATABASE
# ==========================================

def connect_db():
    return sqlite3.connect(str(DB_PATH), timeout=20)

def initialize_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                student_name TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(student_id, attendance_date)
            )
        """)

initialize_db()

# ==========================================
# FIND TRAINED MODEL
# ==========================================

def find_model():
    if not MODEL_DIR.exists():
        return None

    candidates = []

    for ext in ["*.yml", "*.yaml", "*.xml"]:
        for file in MODEL_DIR.glob(ext):
            if "haarcascade" not in file.name.lower():
                candidates.append(file)

    return candidates[0] if candidates else None


def load_labels():
    files = list(MODEL_DIR.glob("*labels*.npy"))

    if not files:
        return {}

    data = np.load(files[0], allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    result = {}

    if isinstance(data, dict):
        for name, label in data.items():
            try:
                result[int(label)] = str(name)
            except (ValueError, TypeError):
                try:
                    result[int(name)] = str(label)
                except (ValueError, TypeError):
                    continue

    elif isinstance(data, (list, tuple, np.ndarray)):
        for index, name in enumerate(data):
            result[index] = str(name)

    return result


def load_recognition_model():
    global RECOGNIZER, FACE_CASCADE
    global LABEL_MAP, MODEL_ERROR

    try:
        model_file = find_model()

        if model_file is None:
            raise FileNotFoundError(
                "No trained face model (.yml/.yaml) found in models/"
            )

        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV face module unavailable. "
                "Install opencv-contrib-python-headless."
            )

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(model_file))

        cascade_file = (
            MODEL_DIR / "haarcascade_frontalface_default.xml"
        )

        if not cascade_file.exists():
            cascade_file = Path(
                cv2.data.haarcascades +
                "haarcascade_frontalface_default.xml"
            )

        cascade = cv2.CascadeClassifier(str(cascade_file))

        if cascade.empty():
            raise RuntimeError("Unable to load face detector.")

        labels = load_labels()

        if not labels:
            raise RuntimeError(
                "Could not load labels.npy. Check your training output."
            )

        RECOGNIZER = recognizer
        FACE_CASCADE = cascade
        LABEL_MAP = labels
        MODEL_ERROR = None

        with connect_db() as conn:
            for label, name in labels.items():
                conn.execute(
                    """INSERT OR IGNORE INTO students
                    (student_id, student_name) VALUES (?, ?)""",
                    (str(name), str(name))
                )

    except Exception as exc:
        MODEL_ERROR = str(exc)

load_recognition_model()

# ==========================================
# ATTENDANCE RECORDING
# ==========================================

def mark_present(student_name):
    now = datetime.now()

    try:
        with connect_db() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO attendance
                (student_id, student_name,
                 attendance_date, attendance_time, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                str(student_name),
                str(student_name),
                now.date().isoformat(),
                now.strftime("%H:%M:%S"),
                "Present"
            ))

    except sqlite3.Error:
        pass

# ==========================================
# WEBCAM PROCESSING
# ==========================================

def process_frame(frame):
    global STATE

    image = frame.to_ndarray(format="bgr24")
    image = cv2.flip(image, 1)

    if RECOGNIZER is None or FACE_CASCADE is None:
        cv2.putText(
            image,
            "Recognition model unavailable",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )
        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        with STATE_LOCK:
            STATE.update({
                "name": "No face detected",
                "confidence": None,
                "status": "Waiting",
                "message": "Position your face in the camera"
            })

    for x, y, w, h in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        try:
            label, confidence = RECOGNIZER.predict(face)

            name = LABEL_MAP.get(int(label))

            if name is not None and confidence <= CONFIDENCE_LIMIT:

                mark_present(name)

                with STATE_LOCK:
                    STATE.update({
                        "name": name,
                        "confidence": round(float(confidence), 2),
                        "status": "PRESENT",
                        "last_seen": datetime.now().strftime(
                            "%H:%M:%S"
                        ),
                        "message": "Face successfully matched"
                    })

                color = (0, 200, 0)
                display = f"{name} - PRESENT"

            else:
                with STATE_LOCK:
                    STATE.update({
                        "name": "Unknown person",
                        "confidence": round(float(confidence), 2),
                        "status": "Not marked",
                        "message": "No verified student match"
                    })

                color = (0, 0, 255)
                display = "Unknown - NOT MARKED"

        except Exception:
            color = (0, 0, 255)
            display = "Recognition error"

        cv2.rectangle(
            image, (x, y), (x+w, y+h), color, 2
        )

        cv2.putText(
            image,
            display,
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    return av.VideoFrame.from_ndarray(
        image, format="bgr24"
    )


# ==========================================
# PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
}
[data-testid="stSidebar"] {
    background: #10243a;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #10243a;
}
.metric-card {
    background: white;
    border-radius: 13px;
    padding: 20px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("# SMART ATTENDANCE")
st.sidebar.caption("INTELLIGENT FACE RECOGNITION")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Live Face Recognition",
        "Student Records",
        "Attendance Reports"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Face Recognition Attendance Platform")

# ==========================================
# DASHBOARD
# ==========================================

def get_data():
    with connect_db() as conn:
        students = pd.read_sql_query(
            "SELECT * FROM students", conn
        )

        attendance = pd.read_sql_query(
            "SELECT * FROM attendance ORDER BY id DESC",
            conn
        )

    return students, attendance


students, attendance = get_data()
today = date.today().isoformat()

today_data = (
    attendance[attendance["attendance_date"] == today]
    if not attendance.empty
    else pd.DataFrame()
)

present_count = (
    len(today_data[today_data["status"] == "Present"])
    if not today_data.empty else 0
)

total_students = len(students)
absent_count = max(total_students - present_count, 0)
rate = (
    round(present_count / total_students * 100, 1)
    if total_students else 0
)

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Smart Attendance Dashboard</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Attendance overview | {datetime.now().strftime('%d %B %Y')}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)
    c2.metric("Present Today", present_count)
    c3.metric("Absent Today", absent_count)
    c4.metric("Attendance Rate", f"{rate}%")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Today's Attendance")

        if today_data.empty:
            st.info("No attendance recorded today.")
        else:
            st.dataframe(
                today_data,
                use_container_width=True,
                hide_index=True
            )

    with right:
        st.subheader("System Status")

        if MODEL_ERROR:
            st.error("Recognition model not ready")
            st.caption(MODEL_ERROR)
        else:
            st.success("Recognition model loaded")

        st.metric("Registered Face Labels", len(LABEL_MAP))

# ==========================================
# LIVE RECOGNITION
# ==========================================

elif page == "Live Face Recognition":

    st.title("Live Face Recognition")
    st.write(
        "Allow browser camera permission to begin student verification."
    )

    if MODEL_ERROR:
        st.error("Face recognition model could not load.")
        st.code(MODEL_ERROR)
        st.info(
            "Check models/ for the trained LBPH model and labels.npy."
        )

    else:
        st.success("Trained recognition model loaded.")

        confidence_limit = st.slider(
            "Recognition confidence threshold "
            "(lower is stricter)",
            min_value=30,
            max_value=100,
            value=70
        )

        CONFIDENCE_LIMIT = float(confidence_limit)

        webrtc_streamer(
            key="smart-attendance-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=process_frame,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.divider()
        st.subheader("Recognition Result")

        with STATE_LOCK:
            current = STATE.copy()

        st.write("Student:", current["name"])

        if current["status"] == "PRESENT":
            st.success("PRESENT - Attendance recorded")

        elif current["status"] == "Not marked":
            st.error("Unknown student - Attendance not recorded")

        else:
            st.warning(current["message"])

        if current["confidence"] is not None:
            st.write(
                "LBPH confidence:",
                current["confidence"]
            )

        if current["last_seen"]:
            st.write("Last recognized:", current["last_seen"])

        st.caption(
            "Attendance is recorded only once per student per day."
        )

# ==========================================
# STUDENT RECORDS
# ==========================================

elif page == "Student Records":

    st.title("Student Management")

    with st.form("student_registration"):
        student_id = st.text_input("Student ID")
        student_name = st.text_input("Student Name")

        submitted = st.form_submit_button("Register Student")

        if submitted:
            if not student_id.strip() or not student_name.strip():
                st.error("Enter both ID and name.")
            else:
                try:
                    with connect_db() as conn:
                        conn.execute(
                            """INSERT INTO students
                            VALUES (?, ?)""",
                            (student_id.strip(), student_name.strip())
                        )
                    st.success(
                        "Student record added. "
                        "A trained face label is still required "
                        "for recognition."
                    )
                    st.rerun()

                except sqlite3.IntegrityError:
                    st.error("Student ID already exists.")

    st.subheader("Registered Students")
    st.dataframe(students, use_container_width=True)

# ==========================================
# ATTENDANCE REPORTS
# ==========================================

elif page == "Attendance Reports":

    st.title("Attendance Reports")

    if attendance.empty:
        st.info("No attendance records available.")

    else:
        st.dataframe(
            attendance,
            use_container_width=True,
            hide_index=True
        )

        csv = attendance.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Attendance CSV",
            csv,
            "smart_attendance_report.csv",
            "text/csv"
        )

        st.subheader("Attendance Summary")

        summary = attendance.groupby(
            "student_name"
        ).size().reset_index(name="Days Present")

        st.dataframe(summary, use_container_width=True)
