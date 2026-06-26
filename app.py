import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.set_page_config(page_title="Emotion Detector", page_icon="🎭")

st.title("🎭 Face Emotion Detector")
st.write("Allow camera → Take a snap → AI detects your emotion!")

EMOJI = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
}

img_file = st.camera_input("📸 Take a photo")

if img_file:
    img = Image.open(img_file).convert("RGB")

    with st.spinner("Analyzing emotion..."):
        try:
            result = DeepFace.analyze(
                np.array(img),
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            result = result if isinstance(result, list) else [result]
            face = result[0]
            dominant = face["dominant_emotion"]
            emotions = face["emotion"]

            st.success(f"{EMOJI.get(dominant, '')} You look **{dominant.upper()}**!")

            st.write("### Emotion Breakdown")
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                st.progress(int(score), text=f"{EMOJI.get(emotion, '')} {emotion} — {score:.1f}%")

        except Exception:
            st.error("No face detected. Try again with better lighting!")

st.markdown("---")
st.caption("Built by Harvi Sharma · B.Tech CSE @ Graphic Era University")
