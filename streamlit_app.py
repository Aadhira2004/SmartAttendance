import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
from pathlib import Path

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📋",
    layout="wide"
)

DB_PATH = Path("attendance_streamlit.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            parent_email TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            attendance_time TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def load_students():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM students", conn)

def load_attendance():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM attendance ORDER BY id DESC", conn
        )

st.sidebar.title("SMART ATTENDANCE")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Register Student", "Mark Attendance",
     "Attendance Records", "About"]
)

students = load_students()
attendance = load_attendance()
today = date.today().isoformat()

if page == "Dashboard":
    st.title("Smart Attendance System")
    st.caption("Face Recognition Based Attendance Management")

    today_records = attendance[
        attendance["attendance_date"] == today
    ] if not attendance.empty else pd.DataFrame()

    present = len(today_records[
        today_records["status"] == "Present"
    ]) if not today_records.empty else 0

    total = len(students)
    absent = max(total - present, 0)
    percentage = round((present / total) * 100, 1) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total)
    c2.metric("Present Today", present)
    c3.metric("Absent Today", absent)
    c4.metric("Attendance Rate", f"{percentage}%")

    st.subheader("Today's Attendance")
    if today_records.empty:
        st.info("No attendance recorded today.")
    else:
        st.dataframe(today_records, use_container_width=True)

    st.subheader("Recent Students")
    st.dataframe(students, use_container_width=True)

elif page == "Register Student":
    st.title("Student Registration")

    with st.form("register_form"):
        student_id = st.text_input("Student ID")
        name = st.text_input("Student Name")
        email = st.text_input("Student Email")
        parent_email = st.text_input("Parent Email")
        submit = st.form_submit_button("Register Student")

        if submit:
            if not student_id.strip() or not name.strip():
                st.error("Student ID and name are required.")
            else:
                try:
                    with get_connection() as conn:
                        conn.execute(
                            """INSERT INTO students
                            VALUES (?, ?, ?, ?)""",
                            (student_id.strip(), name.strip(),
                             email.strip(), parent_email.strip())
                        )
                    st.success("Student registered successfully.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("This Student ID already exists.")

    st.subheader("Registered Students")
    st.dataframe(students, use_container_width=True)

elif page == "Mark Attendance":
    st.title("Attendance Management")
    st.warning(
        "Manual attendance entry only. Face recognition is not "
        "connected in this version."
    )

    if students.empty:
        st.info("Register students first.")
    else:
        student_options = students["student_id"].tolist()
        selected = st.selectbox("Select Student ID", student_options)

        selected_name = students.loc[
            students["student_id"] == selected, "name"
        ].iloc[0]

        st.write("Student:", selected_name)

        if st.button("Mark Present"):
            with get_connection() as conn:
                exists = conn.execute(
                    """SELECT 1 FROM attendance
                    WHERE student_id=? AND attendance_date=?
                    AND status='Present'""",
                    (selected, today)
                ).fetchone()

                if exists:
                    st.warning("Attendance already marked today.")
                else:
                    conn.execute(
                        """INSERT INTO attendance
                        (student_id, attendance_date,
                         attendance_time, status)
                        VALUES (?, ?, ?, ?)""",
                        (selected, today,
                         datetime.now().strftime("%H:%M:%S"),
                         "Present")
                    )
                    st.success("Attendance marked successfully.")
                    st.rerun()

elif page == "Attendance Records":
    st.title("Attendance Reports")

    if attendance.empty:
        st.info("No attendance records available.")
    else:
        st.dataframe(attendance, use_container_width=True)

        csv = attendance.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Attendance CSV",
            data=csv,
            file_name="attendance_report.csv",
            mime="text/csv"
        )

elif page == "About":
    st.title("About Smart Attendance")
    st.write(
        "Smart Attendance is an attendance management platform "
        "designed to support student registration, attendance "
        "tracking and reporting."
    )
    st.caption("Face recognition integration is a separate next step.")
