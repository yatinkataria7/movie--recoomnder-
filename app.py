
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies_cleaned.csv")

# Fit vectorizer once
cv = CountVectorizer(max_features=3000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])

def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return ["Movie not found."]
    idx = movies[movies['title'].str.lower() == movie].index[0]
    vec = vectors[idx]
    sim = cosine_similarity(vec, vectors).flatten()
    sim_scores = list(enumerate(sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sim_scores]

# Streamlit UI
st.title("🎥 Movie Recommender System")

selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    st.write("Top 5 Recommended Movies:")
    recommendations = recommend(selected_movie)
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
