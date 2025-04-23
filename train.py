# train.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load and merge your datasets
movies = pd.read_csv("movies.csv")  # Ensure this CSV is in your repo

# Preprocess data
movies['content'] = movies['title'] + ' ' + movies['genres']  # or any other valid columns
vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(movies['content'].fillna(''))


# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Save the model and data
with open('movie_model.pkl', 'wb') as f:
    pickle.dump(similarity, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('movies_df.pkl', 'wb') as f:
    pickle.dump(movies, f)

print("✅ Model and vectorizer saved successfully.")
