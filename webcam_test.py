import streamlit as st
import av
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(
    page_title="Webcam Test",
    page_icon="📷",
    layout="wide"
)

st.title("Smart Attendance - Webcam Test")
st.write("Testing live browser camera integration.")

def camera_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.flip(img, 1)

    cv2.putText(
        img,
        "WEBCAM WORKING",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )

ctx = webrtc_streamer(
    key="webcam-test",

    mode=WebRtcMode.SENDRECV,

    video_frame_callback=camera_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    async_processing=True
)

if ctx.state.playing:
    st.success("WEBCAM CONNECTED")
else:
    st.warning("Click START to activate the webcam.")
