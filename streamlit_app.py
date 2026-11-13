import streamlit as st
import cv2
import av
import sqlite3
import pandas as pd
import threading
import time
import uuid

from pathlib import Path
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ==================================
# SETTINGS
# ==================================

BASE = Path(__file__).resolve().parent
CASCADE = BASE / "models" / "haarcascade_frontalface_default.xml"
DATABASE = BASE / "anonymous_attendance.db"

st.set_page_config(
    page_title="Smart Attendance",
    page_icon="camera",
    layout="wide"
)

LOCK = threading.Lock()

# ==================================
# DATABASE
# ==================================

def connect():
    return sqlite3.connect(str(DATABASE), timeout=30)


def setup_database():

    con = connect()

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

    con.commit()
    con.close()


setup_database()


def save_attendance(session, person):

    now = datetime.now()

    with LOCK:

        con = connect()

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

        con.commit()
        con.close()


# ==================================
# LOAD FACE DETECTOR
# ==================================

@st.cache_resource
def load_detector():

    if not CASCADE.exists():
        raise FileNotFoundError(
            "Face detector XML file not found."
        )

    detector = cv2.CascadeClassifier(str(CASCADE))

    if detector.empty():
        raise RuntimeError(
            "Unable to load Haar face detector."
        )

    return detector


try:
    detector = load_detector()
    detector_error = None

except Exception as e:
    detector = None
    detector_error = str(e)


# ==================================
# WEBCAM PROCESSOR
# ==================================

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

        # Remove face tracks not seen recently
        self.tracks = {
            key: value for key, value in self.tracks.items()
            if now - value["last_seen"] < 2.0
        }

        for (x, y, w, h) in faces:

            cx = x + w // 2
            cy = y + h // 2

            best_id = None
            best_distance = 100

            # Match this face to an existing nearby track
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

            # A new face track means one new PRESENT event
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
                (0, 200, 0),
                2
            )

            cv2.putText(
                image,
                "PRESENT",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2
            )

        self.current_faces = len(faces)

        cv2.putText(
            image,
            f"Faces: {self.current_faces}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ==================================
# SIDEBAR
# ==================================

st.sidebar.title("SMART ATTENDANCE")
st.sidebar.caption("Anonymous Face Detection")

page = st.sidebar.radio(
    "MENU",
    [
        "Dashboard",
        "Live Attendance",
        "Attendance Reports"
    ]
)


# ==================================
# DASHBOARD
# ==================================

if page == "Dashboard":

    st.title("Smart Attendance Dashboard")

    st.write(
        datetime.now().strftime("%d %B %Y")
    )

    con = connect()

    total = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM attendance",
        con
    ).iloc[0, 0]

    today = datetime.now().strftime("%Y-%m-%d")

    present = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM attendance WHERE date=?",
        con,
        params=(today,)
    ).iloc[0, 0]

    con.close()

    a, b = st.columns(2)

    a.metric("Total Recorded Events", int(total))
    b.metric("Today's PRESENT Events", int(present))

    if detector_error:
        st.error(detector_error)
    else:
        st.success("Face detector ready")

    st.info(
        "These are anonymous face-detection events, "
        "not verified individual student attendance."
    )


# ==================================
# LIVE ATTENDANCE
# ==================================

elif page == "Live Attendance":

    st.title("Live Face Recognition")

    st.write(
        "Show a face to the webcam to trigger an "
        "anonymous PRESENT attendance event."
    )

    st.warning(
        "This system detects faces but does not identify "
        "people. Photos and video replays may also trigger detection."
    )

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
                "Recorded Events This Session",
                processor.total
            )

            b.metric(
                "New Face Tracks",
                processor.total
            )

        else:

            st.info("Press START to open the camera.")


# ==================================
# ATTENDANCE REPORTS
# ==================================

elif page == "Attendance Reports":

    st.title("Attendance Reports")

    con = connect()

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

    con.close()

    st.metric("Total Records", len(data))

    if data.empty:

        st.info("No attendance records available.")

    else:

        st.dataframe(
            data,
            use_container_width=True
        )

        st.download_button(
            "Download CSV Report",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="attendance_report.csv",
            mime="text/csv"
        )
