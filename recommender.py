import os
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvggish import vggish, vggish_input, vggish_params
import torch
import tempfile
import soundfile as sf
from panns_inference import AudioTagging, labels
import librosa
import numpy as np
from pyspark import SparkContext, SparkConf
import dask.dataframe as dd
from dask import delayed
import joblib

# Paths
AUDIO_DIR = '/Users/anchitmulye/Downloads/fma_small'
TRACKS_CSV = '/Users/anchitmulye/Downloads/fma_metadata/tracks.csv'

tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
# track_info = tracks['track'][['genre_top', 'title']].dropna()
track_info = pd.concat([
    tracks['track'][['genre_top', 'title']],
    tracks['album'][['title']].rename(columns={'title': 'album'}),
    tracks['artist'][['name']].rename(columns={'name': 'artist'})
], axis=1).dropna()
track_info.to_csv("track_info.csv")
print(track_info.info())

# Enable for testing
# track_info = track_info.head(100)

# Initialize models
model = vggish()
model.eval()

at = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')


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
        # print(f"Error processing {file_path}: {e}")
        return None


def get_track_path(track_id):
    tid_str = f"{track_id:06d}"
    return os.path.join(AUDIO_DIR, tid_str[:3], f"{tid_str}.mp3")


conf = SparkConf().setAppName("MusicFeatureExtraction")
sc = SparkContext(conf=conf)

file_paths_rdd = sc.parallelize([get_track_path(track_id) for track_id in track_info.index])


def extract_features_rdd(file_path):
    return extract_panns_features(file_path)


# Extract features in parallel using PySpark
features_rdd = file_paths_rdd.map(extract_features_rdd)

# Collect features and valid indices
features = features_rdd.filter(lambda x: x is not None).collect()
valid_indices = [track_info.index[i] for i, feat in enumerate(features) if feat is not None]

# Convert features into a DataFrame
features_df = pd.DataFrame(features, index=valid_indices)
print(features_df.head(10))

# --- Dask Integration ---
# Using Dask to handle large DataFrames efficiently
dask_df = dd.from_pandas(features_df, npartitions=4)

# Standardize features using Dask
scaler = StandardScaler()
features_scaled = scaler.fit_transform(dask_df.compute())
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features_scaled, "features_scaled.pkl")

# Compute similarity matrix using Dask
similarity = cosine_similarity(features_scaled)
similarity_df = pd.DataFrame(similarity, index=features_df.index, columns=features_df.index)
joblib.dump(similarity_df, "similarity_df.pkl")


# --- Recommendation Function ---
def recommend(track_id, top_n=5):
    if track_id not in features_df.index:
        print("Track not found in features.")
        return
    idx = features_df.index.get_loc(track_id)
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = [x for x in sim_scores if features_df.index[x[0]] != track_id]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [(features_df.index[i], score) for i, score in sim_scores[:top_n]]
    print(f"\n🎵 Recommendations for: {track_info.loc[track_id, 'title']} ({track_info.loc[track_id, 'genre_top']})\n")
    for i, (tid, score) in enumerate(top_indices, 1):
        title = track_info.loc[tid, 'title']
        genre = track_info.loc[tid, 'genre_top']
        print(f"{i}. {title} ({genre}) - Similarity: {score:.4f}")


def recall_at_k(k=5, sample_size=100):
    recalls = []
    sample_tracks = features_df.index[:sample_size]

    for track_id in sample_tracks:
        genre = track_info.loc[track_id, 'genre_top']
        artist = track_info.loc[track_id, 'artist']

        # Define relevant tracks
        relevant = set(track_info[
                           (track_info['genre_top'] == genre) | (track_info['artist'] == artist)
                           ].index) - {track_id}

        if not relevant:
            continue  # skip if no known relevant tracks

        # Get recommendations
        idx = features_df.index.get_loc(track_id)
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = [x for x in sim_scores if features_df.index[x[0]] != track_id]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_k = [features_df.index[i] for i, _ in sim_scores[:k]]

        # Calculate recall
        retrieved_relevant = len(set(top_k) & relevant)
        recall = retrieved_relevant / len(relevant)
        recalls.append(recall)

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print(f"\n🔍 Average Recall@{k}: {avg_recall:.4f} over {len(recalls)} samples")
    else:
        print("⚠️ No valid samples found for Recall evaluation.")


# --- Example usage ---
if __name__ == "__main__":
    print("\nCosine Similarity Matrix (Partial View):")
    print(similarity_df.iloc[:5, :5])

    sample_track = features_df.index[3]
    recommend(sample_track, top_n=15)

    print("\nRecall:")
    recall_at_k(k=10, sample_size=50)
