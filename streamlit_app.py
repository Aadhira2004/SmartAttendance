import streamlit as st
import cv2
import av
import sqlite3
import pandas as pd
import threading
import time
import uuid
import hmac
import os

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

BASE = Path(__file__).resolve().parent
CASCADE = BASE / "models" / "haarcascade_frontalface_default.xml"
DATABASE = BASE / "anonymous_attendance.db"
IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(
    page_title="Smart Attendance | Admin",
    page_icon="camera",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOCK = threading.Lock()

# PROFESSIONAL THEME

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FB;
}

[data-testid="stSidebar"] {
    background: #10243A;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
}

h1, h2, h3 {
    color: #10243A;
    font-weight: 700;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2EAF2;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 3px 12px rgba(16,36,58,0.04);
    min-height: 125px;
}

.metric-label {
    color: #66788A;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #10243A;
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.stButton > button,
.stDownloadButton > button {
    background: #087F8C;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 9px 18px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #066873;
    color: white;
    border: none;
}

.login-panel {
    background: white;
    padding: 30px;
    border-radius: 16px;
    border: 1px solid #E2EAF2;
}

hr {
    border-color: #E2EAF2;
}
</style>
""", unsafe_allow_html=True)


# DATABASE

def connect():
    return sqlite3.connect(str(DATABASE), timeout=30)


def setup_database():
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                anonymous_id TEXT,
                date TEXT,
                time TEXT,
                status TEXT
            )
        """)


setup_database()


def save_attendance(session, person):
    now = datetime.now(IST)

    with LOCK:
        with connect() as con:
            con.execute("""
                INSERT INTO attendance
                (session_id, anonymous_id, date, time, status)
                VALUES (?, ?, ?, ?, 'PRESENT')
            """, (
                session,
                person,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S")
            ))


# ADMIN AUTHENTICATION

def get_admin_credentials():
    username = os.getenv("ADMIN_USERNAME", "")
    password = os.getenv("ADMIN_PASSWORD", "")

    try:
        username = st.secrets["admin"]["username"]
        password = st.secrets["admin"]["password"]
    except Exception:
        pass

    return str(username), str(password)


def login_page():
    st.markdown(
        "<div style='text-align:center;padding-top:65px'>"
        "<h1>SMART ATTENDANCE</h1>"
        "<p style='color:#087F8C;font-weight:600'>ADMINISTRATION PORTAL</p>"
        "</div>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown('<div class="login-panel">', unsafe_allow_html=True)
        st.subheader("Admin Login")
        st.write("Sign in to manage the attendance dashboard.")

        username_input = st.text_input("Username")
        password_input = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Sign In", use_container_width=True):
            correct_user, correct_password = get_admin_credentials()

            if not correct_user or not correct_password:
                st.error("Admin credentials are not configured.")
            elif (
                hmac.compare_digest(username_input, correct_user)
                and hmac.compare_digest(password_input, correct_password)
            ):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.get("authenticated", False):
    login_page()
    st.stop()


# LOAD FACE DETECTOR

@st.cache_resource
def load_detector():
    if not CASCADE.exists():
        raise FileNotFoundError(
            "Haar cascade XML file not found in models folder."
        )

    model = cv2.CascadeClassifier(str(CASCADE))

    if model.empty():
        raise RuntimeError("Unable to load face detector.")

    return model


try:
    detector = load_detector()
    detector_error = None
except Exception as error:
    detector = None
    detector_error = str(error)


# WEBCAM PROCESSOR

class CameraProcessor(VideoProcessorBase):

    def __init__(self):
        self.session = str(uuid.uuid4())
        self.tracks = {}
        self.next_person = 1
        self.total = 0
        self.current_faces = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        now = time.time()
        matched = set()

        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if now - value["last_seen"] < 2.0
        }

        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2

            best_id = None
            best_distance = 100

            for person_id, track in self.tracks.items():
                if person_id in matched:
                    continue

                old_x, old_y = track["center"]

                distance = (
                    (cx - old_x) ** 2 +
                    (cy - old_y) ** 2
                ) ** 0.5

                if distance < best_distance:
                    best_distance = distance
                    best_id = person_id

            if best_id is None:
                best_id = f"PERSON_{self.next_person:04d}"
                self.next_person += 1

                save_attendance(self.session, best_id)
                self.total += 1

            self.tracks[best_id] = {
                "center": (cx, cy),
                "last_seen": now
            }

            matched.add(best_id)

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 160, 120),
                2
            )

            cv2.putText(
                image,
                "PRESENT",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 120),
                2
            )

        self.current_faces = len(faces)

        cv2.putText(
            image,
            f"Faces: {self.current_faces}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 160, 120),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# SIDEBAR

st.sidebar.markdown(
    "<h2 style='color:white'>SMART ATTENDANCE</h2>"
    "<p style='color:#9FB4C9'>ADMIN CONTROL PANEL</p>",
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Live Attendance",
        "Attendance Reports"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Administrator session active")

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()


# DASHBOARD

if page == "Dashboard":

    st.title("Attendance Dashboard")

    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    st.caption(
        now.strftime("%A, %d %B %Y | %I:%M %p IST")
    )

    with connect() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM attendance"
        ).fetchone()[0]

        today_present = con.execute("""
            SELECT COUNT(*)
            FROM attendance
            WHERE date = ? AND UPPER(status) = 'PRESENT'
        """, (today,)).fetchone()[0]

        tomorrow_present = con.execute("""
            SELECT COUNT(*)
            FROM attendance
            WHERE date = ? AND UPPER(status) = 'PRESENT'
        """, (tomorrow,)).fetchone()[0]

    st.markdown("### Attendance Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Events</div>
                <div class="metric-value">{total:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Today's PRESENT</div>
                <div class="metric-value">{today_present:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tomorrow's PRESENT</div>
                <div class="metric-value">{tomorrow_present:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("System Status")

    if detector_error:
        st.error(detector_error)
    else:
        st.success("Face detector ready")

    st.caption("Attendance metrics are calculated independently by date.")


# LIVE ATTENDANCE

elif page == "Live Attendance":

    st.title("Live Attendance")

    st.write("Start the camera to detect faces and record attendance events.")

    if detector_error:
        st.error(detector_error)
    else:
        ctx = webrtc_streamer(
            key="smart-attendance-camera",
            video_processor_factory=CameraProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        if ctx.state.playing and ctx.video_processor:
            processor = ctx.video_processor

            st.success("Camera running")

            a, b = st.columns(2)

            a.metric(
                "PRESENT Events This Session",
                processor.total
            )

            b.metric(
                "Faces Currently Visible",
                processor.current_faces
            )
        else:
            st.info("Press START to open the camera.")


# ATTENDANCE REPORTS

elif page == "Attendance Reports":

    st.title("Attendance Reports")

    with connect() as con:
        data = pd.read_sql_query(
            """
            SELECT
                id AS Record_ID,
                session_id AS Session,
                anonymous_id AS Anonymous_Person,
                date AS Attendance_Date,
                time AS Attendance_Time,
                status AS Status
            FROM attendance
            ORDER BY id DESC
            """,
            con
        )

    st.metric("Total Records", f"{len(data):,}")

    if data.empty:
        st.info("No attendance records available.")
    else:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download CSV Report",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="attendance_report.csv",
            mime="text/csv"
        )
