# 🎬 JayCineMatch — Movie Recommendation System

A content-based movie recommendation system built with Python, pandas, scikit-learn, TF-IDF, cosine similarity, and Streamlit.

JayCineMatch recommends movies based on similarities in their plot descriptions, genres, keywords, cast members, and directors. Users can select a movie, choose how many recommendations to receive, and filter results by a minimum similarity score.

## Project Overview

Movie streaming platforms often contain thousands of titles, making it difficult for users to discover movies that match their interests.

JayCineMatch explores this problem by building a recommendation engine that compares movie content and ranks similar films.

**Project type:** Data Science / Machine Learning
**Recommendation approach:** Content-based filtering
**Interface:** Streamlit
**Dataset:** TMDB 5000 Movies and Credits datasets

## Objectives

* Prepare and explore a real-world movie dataset.
* Engineer meaningful features from movie metadata.
* Convert text features into numerical representations using TF-IDF.
* Calculate movie similarity using cosine similarity.
* Experiment with different feature-weighting strategies.
* Build an interactive movie recommendation application.
* Present the project as a reproducible data science workflow.

## Features

* Select a movie from a searchable dropdown.
* Generate between 1 and 20 recommendations.
* Adjust the minimum similarity threshold.
* View recommended movies and similarity scores.
* Compare a baseline TF-IDF model with a weighted-feature experiment.

## Technology Stack

* **Python** — programming language
* **pandas** — data manipulation and cleaning
* **scikit-learn** — TF-IDF vectorization and cosine similarity
* **Joblib** — model artifact persistence
* **Streamlit** — interactive web application
* **Jupyter Notebook / Google Colab** — exploration and model development

## Dataset

The project uses the TMDB 5000 Movies and Credits datasets, which contain movie metadata such as titles, genres, plot descriptions, keywords, cast, and crew information.

The original dataset contains 4,803 movies and 4,803 corresponding credits records.

Dataset source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Please review the dataset's applicable terms and attribution requirements before redistributing its files.

## Data Science Workflow

The project follows this workflow:

1. **Problem Definition** — Define the movie discovery problem.
2. **Data Collection** — Load the movies and credits datasets.
3. **Data Cleaning** — Handle missing plot descriptions and inspect duplicates.
4. **Exploratory Data Analysis** — Examine movie ratings, vote counts, and genre distributions.
5. **Feature Engineering** — Extract genres, keywords, the top three cast members, and directors.
6. **Model Selection** — Choose a content-based recommendation approach.
7. **Model Training** — Fit a TF-IDF vectorizer to movie features.
8. **Hyperparameter Tuning** — Experiment with feature weighting and vectorizer settings.
9. **Model Evaluation** — Inspect recommendations for representative movies.
10. **Deployment** — Build an interactive Streamlit application.
11. **Monitoring** — Future work: collect feedback and track recommendation quality.
12. **Retraining** — Future work: update the model when the dataset or feature design changes.

## How the Recommendation Engine Works

### 1. Feature Engineering

The project combines the following information into a text representation for each movie:

* Plot overview
* Genres
* Keywords
* Top three cast members
* Director

Multiword features are normalized, for example, `science fiction` becomes `science_fiction`.

### 2. TF-IDF Vectorization

TF-IDF converts the movie text into numerical feature vectors. Words that are more informative within the dataset can contribute differently from words that occur frequently across many documents.

### 3. Cosine Similarity

The system calculates cosine similarity between the selected movie's vector and the vectors of all other movies.

Movies are ranked by similarity, the selected movie itself is excluded, and results below the configured threshold are removed.

### 4. Feature-Weighting Experiment

The project compares a baseline TF-IDF model with an experimental version that repeats selected structured features.

The experiment showed that weighting changes the ranking and similarity scores. This does not, by itself, prove better recommendation quality; more systematic evaluation is needed.

## Example Results

During development, the weighted model produced these recommendations:

| Input movie     | Example recommendations                              |
| --------------- | ---------------------------------------------------- |
| Avatar          | Aliens, Star Trek Into Darkness, Falcon Rising       |
| The Dark Knight | The Dark Knight Rises, Batman Begins, Batman Returns |
| Titanic         | In the Heart of the Sea, Poseidon, Ghost Ship        |

These are illustrative outputs from the development dataset, not a formal benchmark. Relevance is subjective, and the model does not estimate the probability that a viewer will enjoy a movie.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd JayCineMatch
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the model artifacts

Place the following files in the same directory as `app.py`, unless you update the application paths:

* `tfidf_weighted.joblib`
* `tfidf_weighted_matrix.joblib`
* `movies_rec.pkl`

These artifacts were generated during model development. If they are not included in the repository, generate or obtain them using the project's notebook and documented workflow.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Project Limitations

* Recommendations depend on the quality and completeness of the metadata.
* The model does not learn directly from user ratings, clicks, or viewing history.
* Similarity scores are not probabilities, ratings, or measures of user satisfaction.
* A manually selected similarity threshold may exclude useful recommendations.
* Repeating features is a heuristic and does not guarantee improved relevance.
* Evaluation so far is qualitative rather than a formal offline benchmark.

## Future Improvements

* Create a labeled evaluation set and measure precision at K and recall at K.
* Evaluate recommendation diversity and catalog coverage.
* Compare TF-IDF with alternative text representations.
* Tune the influence of genres, keywords, cast, directors, and plot descriptions systematically.
* Add movie posters and additional movie information.
* Add automated tests and input validation.
* Deploy the application to a public hosting platform.
* Collect optional user feedback to evaluate real-world usefulness.

## Skills Demonstrated

Python, pandas, exploratory data analysis, data cleaning, feature engineering, natural language processing, TF-IDF, cosine similarity, recommendation systems, model persistence, Streamlit, and Git/GitHub.

## kingsley Ijeoma

**kingsley Ijeoma**

Certified Data Scientist | Python | Data Analysis | Machine Learning

GitHub: [My Github link]((https://github.com/kayDtechie3))


LinkedIn: [My linkedin](www.linkedin.com/in/kingsley-ijeoma-csc-855b8b1a9)
