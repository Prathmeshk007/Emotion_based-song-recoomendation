import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Setup page config
st.set_page_config(page_title="Emotion Song Recommender", page_icon="🎭")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.markdown("# 🎭 📸 Emotion Based Song Recommendation System")
st.markdown("---")

@st.cache_resource
def load_model():
    # Load the model once and cache it to save memory/time
    return tf.keras.models.load_model('model.h5')

model = load_model()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.sidebar.markdown("## 🎭 Emotion Detection")
option = st.sidebar.radio("Select one option:", ["📁 Browse Image", "📸 Live Emotion"])

# Store detected emotion in session state to keep it across reruns
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None

if option == "📁 Browse Image":
    st.sidebar.write("Upload an image for emotion detection")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image")
        
        if st.sidebar.button("Detect Emotion"):
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
                face_roi = np.expand_dims(np.expand_dims(face_roi, axis=0), axis=-1)
                prediction = model.predict(face_roi, verbose=0)[0]
                emotion = EMOTIONS[np.argmax(prediction)]
                st.session_state.detected_emotion = emotion
                st.markdown(f"### 🎯 Detected Emotion: {emotion.upper()}")
            else:
                st.warning("😔 No face detected in the image.")

elif option == "📸 Live Emotion":
    st.sidebar.write("Use your webcam for real-time detection")
    # Use st.camera_input for Streamlit Cloud compatibility
    img_file = st.camera_input("Take a snapshot")
    
    if img_file:
        image = Image.open(img_file).convert('RGB')
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
            face_roi = np.expand_dims(np.expand_dims(face_roi, axis=0), axis=-1)
            prediction = model.predict(face_roi, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(prediction)]
            st.session_state.detected_emotion = emotion
            st.success(f"🎯 Detected Emotion: {emotion.upper()}")
        else:
            st.warning("No face detected. Try adjusting your position.")

st.markdown("---")
st.markdown("## 🎵 Song Recommendation")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Preferences")
    lang = st.text_input("Enter language (e.g. Hindi, English)")
    singer = st.text_input("Enter singer name (optional)")

with col2:
    st.markdown("### Get Songs")
    if st.session_state.detected_emotion and lang:
        em = st.session_state.detected_emotion
        search_query = f"{lang} {em} songs {singer}".strip().replace(" ", "+")
        youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
        
        st.write(f"Click below to find **{em}** songs in **{lang}**:")
        st.link_button("🎶 Open YouTube Recommendations", youtube_url)
    else:
        st.info("Please detect an emotion and enter a language first.")