import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from panns_inference import AudioTagging
import torch
import tempfile

# Load required files
features_scaled = joblib.load("features_scaled.pkl")
scaler = joblib.load("scaler.pkl")
similarity_df = joblib.load("similarity_df.pkl")
track_info = pd.read_csv("track_info.csv", index_col=0)

# Load audio embedding model
at = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')


def recommend_by_id(track_id, top_n=10):
    sim_scores = similarity_df.loc[track_id].drop(track_id)
    top_indices = sim_scores.sort_values(ascending=False).head(top_n).index
    recommendations = track_info.loc[top_indices][['title', 'genre_top', 'artist', 'album']]
    return recommendations


def extract_panns_features(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=32000, mono=True)
        waveform = waveform[None, :]
        taggings = at.inference(waveform)
        if isinstance(taggings, tuple):
            embedding = taggings[0]
        else:
            embedding = taggings['embedding']
        return embedding[0]
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None


def recommend_by_audio(uploaded_file, top_n=10):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    embedding = extract_panns_features(tmp_path)
    if embedding is None:
        return None
    if not hasattr(scaler, "mean_"):
        scaler.fit(features_scaled)

    embedding_scaled = scaler.transform([embedding])
    similarities = cosine_similarity(embedding_scaled, features_scaled)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_track_ids = similarity_df.index[top_indices]
    recommendations = track_info.loc[top_track_ids][['title', 'genre_top', 'artist', 'album']]
    return recommendations


# Streamlit UI
st.title("🎧 Music Recommender")

tab1, tab2 = st.tabs(["🔍 Recommend from Song List", "🎵 Upload Audio File"])

with tab1:
    st.header("Select a song to get similar recommendations")
    track_options = {f"{row['title']} ({row['artist']})": idx for idx, row in track_info.iterrows()}
    selected_song = st.selectbox("Choose a track:", list(track_options.keys()))

    if st.button("Get Recommendations"):
        track_id = track_options[selected_song]
        recs = recommend_by_id(track_id)
        st.subheader("Top 10 Recommendations:")
        st.dataframe(recs)

with tab2:
    st.header("Upload an audio file (MP3/WAV)")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        if st.button("Recommend Similar Songs"):
            recs = recommend_by_audio(uploaded_file)
            if recs is not None:
                st.subheader("Top 10 Recommendations:")
                st.dataframe(recs)
