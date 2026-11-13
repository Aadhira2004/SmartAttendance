import streamlit as st

st.set_page_config(
    page_title="Smart Attendance",
    page_icon="📋",
    layout="wide"
)

st.title("Smart Attendance System")
st.subheader("Face Recognition Based Attendance Management")

st.success("Smart Attendance application is running.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Dashboard", "About"]
)

if page == "Dashboard":
    st.header("Attendance Dashboard")
    st.info(
        "Your Streamlit deployment is working. "
        "Face recognition and attendance database integration "
        "will be added next."
    )

elif page == "About":
    st.header("About Smart Attendance")
    st.write(
        "A face-recognition-based attendance management system."
    )
