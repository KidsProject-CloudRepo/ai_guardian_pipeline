import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import altair as alt
import tempfile
import os
import cv2
import speech_recognition as sr
import json

st.set_page_config(page_title="AI Guardian – Emotion Detector", layout="centered")
st.title("😃 AI Guardian – Unified Emotion Detection App (Pipeline Based)")

@st.cache_resource
def load_models():
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    sentiment_model = pipeline("sentiment-analysis", return_all_scores=False)
    image_emotion_model = pipeline("image-classification", model="nateraw/bert-base-uncased-emotion")
    return emotion_model, sentiment_model, image_emotion_model

emotion_model, sentiment_model, image_emotion_model = load_models()

mode = st.selectbox("Select Input Mode:", ["Text", "Image", "Video", "Audio (Upload)", "Audio (Mic)"])

if mode == "Text":
    text_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text"):
        if text_input:
            emotion = emotion_model(text_input)[0]["label"]
            sentiment = sentiment_model(text_input)[0]["label"].lower()
            if emotion in ["sadness", "anger", "disgust"] and sentiment == "positive":
                blended = "joy"
            elif emotion == "joy" and sentiment == "negative":
                blended = "neutral"
            else:
                blended = emotion
            st.success(f"💬 Emotion: **{emotion}** | Sentiment: **{sentiment}** | Final: **{blended}**")

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_image.read())
        image_path = tfile.name
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            try:
                image = Image.open(image_path).convert("RGB")
                predictions = image_emotion_model(image)
                top_pred = predictions[0]
                st.success(f"🖼️ Detected Emotion: **{top_pred['label']}** ({top_pred['score']:.2f})")
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        st.video(video_path)
        if st.button("Analyze Video"):
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            emotion_log = {}
            st.info("Processing video, this may take a moment...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count > 100:
                    break
                try:
                    if frame_count % 10 == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        preds = image_emotion_model(pil_image)
                        emotion = preds[0]["label"]
                        emotion_log[frame_count] = emotion
                except Exception:
                    emotion_log[frame_count] = "error"
                frame_count += 1
            cap.release()
            st.success("Video processing complete!")
            st.write("Sample Detected Emotions:")
            st.json(emotion_log)
            df = pd.DataFrame(list(emotion_log.items()), columns=["Frame", "Emotion"])
            df["Frame"] = df["Frame"].astype(int)
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Frame:O", title="Frame Number"),
                y=alt.Y("count():Q", title="Emotion Count"),
                color="Emotion:N",
                tooltip=["Frame", "Emotion"]
            ).properties(title="📊 Detected Emotions Across Video Frames", width=600, height=300)
            st.altair_chart(chart, use_container_width=True)

elif mode == "Audio (Upload)":
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_file.write(uploaded_audio.read())
            audio_path = audio_file.name
        if st.button("Analyze Audio File"):
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                st.write(f"🗣️ Transcribed Text: _{text}_")
                result = emotion_model(text)
                emotion = result[0]["label"]
                st.success(f"🎧 Detected Emotion from Audio: **{emotion}**")
            except Exception as e:
                st.error(f"Speech Recognition Error: {str(e)}")

elif mode == "Audio (Mic)":
    st.write("🎙️ Click the button and speak after the prompt appears.")
    if st.button("Record from Mic and Analyze"):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                st.info("Listening... please speak clearly.")
                audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.write(f"🗣️ Transcribed Text: _{text}_")
            result = emotion_model(text)
            emotion = result[0]["label"]
            st.success(f"🎤 Detected Emotion from Mic: **{emotion}**")
        except Exception as e:
            st.error(f"Microphone Error: {str(e)}")
