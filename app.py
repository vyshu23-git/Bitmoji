import streamlit as st
import cv2
from emotion_detector import detect_emotion
from PIL import Image
import os

st.title("Emotion → Avatar Generator")

# Avatar mapping
avatar_paths = {
    "happy": "avatars/happy.png",
    "sad": "avatars/sad.png",
    "angry": "avatars/angry.png",
    "surprised": "avatars/surprised.png",
    "neutral": "avatars/neutral.png"
}

# Webcam
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
AVATAR_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:

    ret, frame = camera.read()

    if not ret:
        st.write("Camera error")
        break

    frame, emotion = detect_emotion(frame)

    FRAME_WINDOW.image(frame, channels="BGR")

    avatar_path = avatar_paths.get(emotion, avatar_paths["neutral"])
    avatar = Image.open(avatar_path)

    AVATAR_WINDOW.image(avatar, width=200)

camera.release()
