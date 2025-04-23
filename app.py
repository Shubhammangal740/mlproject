# app.py

from flask import Flask, render_template, request
import os
import pickle

app = Flask(__name__)

# Check if model file exists, if not run training
if not os.path.exists("movie_model.pkl"):
    print("🚀 No model found. Training the model...")
    import train  # will run train.py to generate movie_model.pkl

# Load the model components: cosine_sim, movies df, and title-to-index map
with open("movie_model.pkl", "rb") as f:
    cosine_sim, movies, indices = pickle.load(f)

# Recommendation function
def get_recommendations(title):
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    movie_titles = movies["title"].tolist()
    if request.method == "POST":
        movie_title = request.form["movie"]
        recommendations = get_recommendations(movie_title)
    return render_template("index.html", recommendations=recommendations, movies=movie_titles)

if __name__ == "__main__":
    app.run(debug=True)
