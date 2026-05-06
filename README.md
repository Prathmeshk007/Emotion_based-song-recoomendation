# 🎭 Emotion-Based Song Recommender

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square)](https://tensorflow.org/)

Detect facial emotions and get matching YouTube song recommendations!

## 🚀 Features
- **Live Detection:** Real-time webcam emotion analysis.
- **Image Upload:** Analyze mood from JPG/PNG files.
- **Smart Search:** Searches YouTube by Emotion + Language + Singer.
- **7 Emotions:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## 🛠️ Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/your-username/emotions_song_recom.git
cd emotions_song_recom

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run
```bash
streamlit run app.py
```

## 📂 Structure
- `app.py`: Main Streamlit application.
- `model.h5`: Trained CNN model (16.4M parameters).
- `haarcascade_...xml`: OpenCV face detection model.
- `train/` & `validation/`: FER dataset directories.

## 🧠 Model Info
- **Type:** Sequential CNN
- **Input:** 48x48 Grayscale
- **Output:** 7 Emotion Classes
- **Optimizer:** Adam (Categorical Crossentropy)

## 🔧 Troubleshooting
- **No Webcam?** Check permissions or close other camera apps.
- **Errors?** Ensure `model.h5` and `haarcascade...xml` are in the root folder.
- **Missing Deps?** Run `pip install tensorflow opencv-python streamlit`.

---
<p align="center">Made with ❤️ for Music & AI Lovers</p>