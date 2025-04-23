# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

movies = pd.read_csv("movies.csv").head(1000)
movies['content'] = movies['title'] + ' ' + movies['genres']
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(movies['content'].fillna(''))
cosine_sim = cosine_similarity(vectors)


# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Save all 3 objects together
with open("movie_model.pkl", "wb") as f:
    pickle.dump((cosine_sim, movies, indices), f)

print("✅ Model and vectorizer saved successfully.")
