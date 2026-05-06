import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import webbrowser
from PIL import Image

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.markdown("# 🎭 📸  Emotion base  🎶 song reccomended sysytem")
st.markdown("---")

def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if model:
    st.sidebar.markdown("## 🎭 Emotion Detection")
    option = st.sidebar.radio("Select one option:", ["📁 Browse Image", "📸 Live Emotion"],index=1)
    
    if option == "📁 Browse Image":
        st.sidebar.write("Upload an image for emotion detection")
        uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image")
            
            if st.sidebar.button("Detect Emotion"):
                st.markdown("## 🎭 Detection Results")
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    img_with_rect = img_array.copy()
                    cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    st.image(img_with_rect, caption="Detected Face")
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
                    face_roi = np.expand_dims(np.expand_dims(face_roi, axis=0), axis=-1)
                    prediction = model.predict(face_roi, verbose=0)[0]
                    emotion = EMOTIONS[np.argmax(prediction)]
                    st.session_state.detected_emotion = emotion
                    st.markdown(f"### 🎯 Detected Emotion: {emotion.upper()}")
                    st.markdown("---")
                else:
                    st.warning("😔 No face detected in the image. Please try another image.")
    elif option == "📸 Live Emotion":
        st.sidebar.write("Use webcam for live emotion detection")
        if st.sidebar.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Live Feed")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
                        face_roi = np.expand_dims(np.expand_dims(face_roi, axis=0), axis=-1)
                        prediction = model.predict(face_roi, verbose=0)[0]
                        emotion = EMOTIONS[np.argmax(prediction)]
                        st.session_state.detected_emotion = emotion
                        st.image(frame_rgb, caption="Detected Emotion")
                        st.markdown(f"### Emotion: {emotion.upper()}")
                        st.markdown("---")
                    else:
                        st.warning("No face detected")
                cap.release()
st.markdown("---")
st.markdown("## 🎵 Song Recommendation Based on Emotion")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Enter Singer Details")
    x = st.text_input("Enter language")
    y=st.text_input("Enter singer name")
with col2:
    st.markdown("### Get Recommendations")
    
    if st.button("Search Songs"):
        if 'detected_emotion' in st.session_state and x:
            em = st.session_state.detected_emotion
            webbrowser.open(f"https://www.youtube.com/results?search_query={x}+{em}+songs")
            st.success(f"Searching for {em} {x} songs...")
