import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt
import tempfile
import os
import json
import speech_recognition as sr

st.set_page_config(page_title="AI Guardian – Emotion Detector", layout="centered")
st.title("😃 AI Guardian – Unified Emotion Detection App")

@st.cache_resource
def load_models():
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    sentiment_model = pipeline("sentiment-analysis", return_all_scores=False)
    return emotion_model, sentiment_model

emotion_model, sentiment_model = load_models()

mode = st.selectbox("Select Input Mode:", ["Text", "Audio (Upload)", "Audio (Mic)"])

if mode == "Text":
    text_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text") and text_input:
        emotion = emotion_model(text_input)[0]["label"]
        sentiment = sentiment_model(text_input)[0]["label"].lower()
        st.success(f"💬 Emotion: **{emotion}** | Sentiment: **{sentiment}**")

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
