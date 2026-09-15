# Assignment 20: Building & Deploying a Recommendation System

**Name:** Keval  
**Assignment:** 20  
**System:** Content-Based Movie Recommendation System

## Dataset

**Dataset Name:** MovieLens Latest Full  
**Source:** Kaggle / GroupLens  
**Kaggle Link:** https://www.kaggle.com/datasets/grouplens/movielens-latest-full/data

The attached `movies.csv` contains:

- `movieId` — unique MovieLens identifier
- `title` — movie title
- `genres` — text-based content feature

Dataset size used in this project:


58,098 movies
3 source columns

---

# PART 1 - Data Preprocessing

## Task 1: Load & Understand Dataset

The dataset is loaded with:

```python
pd.read_csv("movies.csv")
```

The console assignment script prints:

- Dataset shape
- Column names
- First 5 rows
- Data types
- Missing values
- Duplicate-title count
- Unique movie IDs
- Unique movie titles
- Unique genre combinations

### Recommendation columns

**Item identifier**

title

**Text feature**

genres

MovieLens Latest Full `movies.csv` does not contain movie overview text in this file, so `genres` is used as the allowed text-based content feature.

---

## Task 2: Text Preprocessing

The genre text is cleaned with these steps:

1. Missing values are replaced with an empty string.
2. Text is converted to lowercase.
3. The `|` genre separator is converted to a space.
4. Punctuation and special characters are removed.
5. English stopwords are removed.
6. The result is stored in `clean_text`.

Example:


Adventure|Animation|Children|Comedy|Fantasy


becomes:


adventure animation children comedy fantasy


Movies containing:


(no genres listed)


are converted to empty content text because there is no real genre information available.

---

# PART 2 - Text Vectorization

## Task 3: TF-IDF

The project uses:

```python
TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2)
)
```

### Why these parameters?

`max_features=500`

Limits the potential vocabulary size and keeps the deployed system compact.

`ngram_range=(1,2)`

Creates both:

- Unigrams such as `comedy`
- Bigrams such as `comedy drama`

The program prints:

- TF-IDF matrix shape
- Vocabulary size
- Sample feature names

---

# Task 4 - Cosine Similarity

Cosine similarity measures the angle between two vectors.

For two vectors `A` and `B`:


cosine_similarity(A, B)


is high when they point in similar directions.

It is useful for TF-IDF because text vectors are sparse and their raw magnitudes are usually less important than the direction/pattern of their features.

## Important Memory-Safe Implementation

The attached MovieLens file contains **58,098 movies**.

A direct dense matrix would have:


58,098 × 58,098


entries.

At float64 precision this would require roughly **27 GB of RAM**, which is inappropriate for a small local or Render deployment.

However, the only content feature in this dataset is `genres`. There are only about 1,600 distinct cleaned genre profiles.

Movies with exactly the same cleaned genre profile receive the same TF-IDF vector.

Therefore the project:

1. Fits TF-IDF on the **entire movie dataset**.
2. Finds unique cleaned genre profiles.
3. Transforms those profiles using the fitted TF-IDF vectorizer.
4. Stores a cosine similarity matrix between unique profiles.
5. Maps every movie back to its profile.

For movies `i` and `j`:


movie_similarity(i, j)
=
profile_similarity(profile_i, profile_j)


This is mathematically equivalent for the dataset's genre-only content representation while avoiding a huge redundant matrix.

---

# PART 3 - Recommendation Logic

## Task 5: `recommend()`

The required function is implemented in `recommender.py`:

```python
def recommend(item_name, bundle, top_n=5):
```

It performs:

1. Find the movie index.
2. Find its genre-content profile.
3. Retrieve cosine-similarity scores.
4. Map scores back to all movies.
5. Sort scores from highest to lowest.
6. Exclude the selected movie itself.
7. Return the top N distinct movie titles.

The console validation tests at least three movies:


Toy Story (1995)
Jumanji (1995)
Matrix, The (1999)


---

# PART 4 - Streamlit Interface

## Task 6

`app.py` provides a simple Streamlit interface with:

- Movie dropdown
- Number-of-recommendations slider
- Recommendation button
- Recommendation table
- Similarity percentage
- Genres

Run locally:

```bash
streamlit run app.py
```

---

# PART 5 - Git & GitHub

## Task 7: Git Setup

Inside the project folder run:

```bash
git init
git add .
git commit -m "Assignment 20 movie recommendation system"
git branch -M main
```

Then connect and push:

```bash
git remote add origin https://github.com/keval1802/GenAI-Task20-Keval.git
git push -u origin main
```

The repository includes all required deployment files:


app.py
requirements.txt
README.md
movies.csv
recommender.py
render.yaml


---

# PART 6 - Render Deployment

## Task 8

### 1. Push the project to GitHub

Make sure the GitHub repository contains the files in this project.

### 2. Open Render

Create a new **Web Service** and connect the GitHub repository.

### 3. Configuration

Use:

**Language**

Python


**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

A `render.yaml` file is also included with this configuration.

### 4. Deploy

Create the service and wait for the deployment to become live.

Render will provide a URL similar to:


https://your-service-name.onrender.com


---

# Task 9 - Final Validation

After Render deployment:

1. Open the Render URL.
2. Select `Toy Story (1995)`.
3. Click **Get Recommendations**.
4. Confirm recommendations appear.
5. Test at least two more movies.
6. Copy the final public URL below.

## Deployed URL


ADD YOUR FINAL RENDER URL HERE AFTER DEPLOYMENT


The URL cannot be pre-filled before the user's GitHub repository and Render account are connected and a successful deployment exists.

---

# Run Streamlit App

```bash
streamlit run app.py
```

---

# Assignment Checklist

- [x] MovieLens Latest Full clearly named
- [x] Kaggle link included
- [x] Attached CSV loaded with pandas
- [x] Dataset shape printed
- [x] Columns printed
- [x] First 5 rows printed
- [x] Essential details printed
- [x] `title` used as item identifier
- [x] `genres` used as text feature
- [x] Lowercase conversion
- [x] Punctuation/special-character removal
- [x] Stopword removal
- [x] Missing-value handling
- [x] `clean_text` column
- [x] TF-IDF
- [x] `max_features`
- [x] `ngram_range`
- [x] TF-IDF matrix shape displayed
- [x] Cosine similarity
- [x] Similarity matrix stored efficiently
- [x] `recommend()` function
- [x] Three recommendation tests
- [x] Streamlit dropdown
- [x] Streamlit recommendation button
- [x] Git-ready files
- [x] `app.py`
- [x] `requirements.txt`
- [x] `README.md`
- [x] Render configuration
- [x] No collaborative filtering
