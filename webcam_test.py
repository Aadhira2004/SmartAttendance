import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2

st.set_page_config(page_title="Smart Attendance", layout="wide")
st.title("Smart Attendance - Webcam Test")

st.write("Allow camera permission when your browser asks.")

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    image = cv2.flip(image, 1)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="attendance-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
