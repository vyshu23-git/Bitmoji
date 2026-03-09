import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pretrained emotion model
model = load_model("emotion_model.hdf5")

# Emotion labels
emotion_labels = ["angry", "happy", "neutral", "sad", "surprised"]

# Face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_emotion(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    emotion = "neutral"

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    return frame, emotion
