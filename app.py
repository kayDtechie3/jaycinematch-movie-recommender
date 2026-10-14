
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Configure the web page
# --------------------------------------------------
st.set_page_config(
    page_title="JayCineMatch - Movie Recommender",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# 2. Load the saved model files
# --------------------------------------------------
BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    vectorizer = joblib.load(
        BASE_DIR / "tfidf_weighted.joblib"
    )

    matrix = joblib.load(
        BASE_DIR / "tfidf_weighted_matrix.joblib"
    )

    movies = pd.read_pickle(
        BASE_DIR / "movies_rec.pkl"
    )

    return vectorizer, matrix, movies


tfidf_weighted, tfidf_weighted_matrix, movies_rec = load_model()


# --------------------------------------------------
# 3. Display the application heading
# --------------------------------------------------
st.title("🎬 JayCineMatch")
st.subheader("Your personal movie discovery assistant")

st.markdown(
    """
    Discover movies based on their genres, plot descriptions,
    keywords, cast, and directors.

    **How to use:** Select a movie, adjust your preferences,
    and click the button to discover similar films.
    """
)

st.divider()


# --------------------------------------------------
# 4. Create the movie selection control
# --------------------------------------------------
movie_titles = sorted(
    movies_rec["title_movie"].dropna().unique().tolist()
)

selected_movie = st.selectbox(
    "Choose a movie",
    movie_titles
)


# --------------------------------------------------
# 5. Create recommendation settings
# --------------------------------------------------
top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=10
)

min_similarity = st.slider(
    "Minimum similarity score",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05
)


# --------------------------------------------------
# 6. Generate recommendations
# --------------------------------------------------
if st.button("Recommend Movies", type="primary"):

    matches = movies_rec[
        movies_rec["title_normalized"]
        == selected_movie.strip().casefold()
    ]

    if matches.empty:
        st.error("Movie not found.")

    else:
        movie_index = matches.index[0]

        # Calculate similarity against every movie
        scores = cosine_similarity(
            tfidf_weighted_matrix[movie_index],
            tfidf_weighted_matrix
        ).flatten()

        # Rank movies from highest to lowest similarity
        ranked_indices = scores.argsort()[::-1]

        # Exclude the selected movie and apply threshold
        ranked_indices = [
            i for i in ranked_indices
            if i != movie_index
            and scores[i] >= min_similarity
        ][:top_n]

        if not ranked_indices:
            st.warning(
                "No movies meet this threshold. "
                "Try lowering the minimum similarity."
            )

        else:
            # Build the recommendation results table
            results = movies_rec.iloc[ranked_indices][
                ["title_movie"]
            ].copy()

            results = results.rename(
                columns={"title_movie": "Movie"}
            )

            results["Similarity Score"] = [
                round(float(scores[i]), 3)
                for i in ranked_indices
            ]

            # Display the recommendation heading
            st.subheader(
                f"Movies similar to {selected_movie}"
            )

            # Display recommendation metrics
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Movies recommended",
                    len(results)
                )

            with col2:
                st.metric(
                    "Highest similarity",
                    f"{results['Similarity Score'].max():.3f}"
                )

            # Display the results table
            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                f"Showing {len(results)} recommendations. "
                "Similarity scores are not ratings or probabilities."
            )