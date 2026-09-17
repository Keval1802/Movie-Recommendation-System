import numpy as np
import pandas as pd

from recommender import load_and_prepare_data, build_recommender, recommend

print("=" * 82)
print("ASSIGNMENT 20 - CONTENT-BASED MOVIE RECOMMENDATION SYSTEM")
print("=" * 82)

print("\n" + "=" * 82)
print("TASK 1 - LOAD & UNDERSTAND DATASET")
print("=" * 82)

df = load_and_prepare_data("movies.csv")

print("\nDataset Name: MovieLens Latest Full")
print("Kaggle Link:")
print("https://www.kaggle.com/datasets/grouplens/movielens-latest-full/data")

print("\nDataset Shape:", df.shape)
print("Column Names:", df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head().to_string(index=False))

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicate Titles:", int(df["title"].duplicated().sum()))
print("Unique Movie IDs:", df["movieId"].nunique())
print("Unique Titles:", df["title"].nunique())
print("Unique Raw Genre Combinations:", df["genres"].nunique())
print("Unique Clean Genre Profiles:", df["clean_text"].nunique())

print("\nText column used for recommendation: genres")
print("Item identifier used: title")

print("\n" + "=" * 82)
print("TASK 2 - TEXT PREPROCESSING")
print("=" * 82)

print("\nCleaning steps:")
print("1. Missing values -> empty string")
print("2. Lowercase conversion")
print("3. Replace MovieLens '|' separators with spaces")
print("4. Remove punctuation / special characters")
print("5. Remove English stopwords")
print("6. Store result in clean_text")

print("\nOriginal Genres vs Clean Text:")
print(
    df[["genres", "clean_text"]]
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 82)
print("TASK 3 - TF-IDF VECTORIZATION")
print("=" * 82)

bundle = build_recommender(df)

tfidf_matrix = bundle["tfidf_matrix"]
vectorizer = bundle["vectorizer"]

print("\nTF-IDF Parameters:")
print("max_features = 500")
print("ngram_range = (1, 2)")

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Vocabulary Size:", len(vectorizer.get_feature_names_out()))

print("\nSample TF-IDF Features:")
print(vectorizer.get_feature_names_out()[:30].tolist())

print("\n" + "=" * 82)
print("TASK 4 - COSINE SIMILARITY")
print("=" * 82)

n_movies = len(df)
dense_bytes = n_movies * n_movies * 8
dense_gb = dense_bytes / (1024 ** 3)

similarity_matrix = bundle["similarity_matrix"]

print(f"\nMovies: {n_movies:,}")
print(
    "Full dense item-item matrix estimate: "
    f"{dense_gb:.2f} GB (float64)"
)

print(
    "Unique cleaned content profiles:",
    len(bundle["unique_profiles"])
)
print(
    "Stored profile similarity matrix shape:",
    similarity_matrix.shape
)

print("\nWhy cosine similarity?")
print("Cosine similarity compares the angle between TF-IDF vectors rather than their")
print("raw magnitude. It works well for sparse text vectors and measures how similar")
print("two movies' genre-feature directions are.")

print("\nMemory note:")
print("A 58k x 58k dense matrix is not practical for a small web deployment.")
print("Because genres are the only text feature, movies with the same clean_text")
print("have identical TF-IDF vectors. The program stores similarity between unique")
print("genre profiles and maps every movie back to a profile. This preserves the")
print("same pairwise genre-based cosine scores without storing redundant rows.")

print("\n" + "=" * 82)
print("TASK 5 - RECOMMENDATION FUNCTION")
print("=" * 82)

test_movies = [
    "Toy Story (1995)",
    "Jumanji (1995)",
    "Matrix, The (1999)",
]

for movie in test_movies:
    print(f"\nTop 5 recommendations for: {movie}")
    results = recommend(movie, bundle, top_n=5)

    if results.empty:
        print("No usable genre information for this movie.")
    else:
        display = results.copy()
        display["similarity"] = display["similarity"].round(4)
        print(display.to_string(index=False))

print("\n" + "=" * 82)
print("TASK 6 - STREAMLIT UI")
print("=" * 82)
print("\nImplemented in app.py:")
print("- searchable dropdown / selectbox")
print("- recommendation count selector")
print("- button")
print("- clear recommendation table")

print("\n" + "=" * 82)
print("TASK 7 - GIT & GITHUB")
print("=" * 82)
print("\nProject contains:")
print("- app.py")
print("- recommender.py")
print("- requirements.txt")
print("- README.md")
print("- render.yaml")
print("- movies.csv")
print("\nGit commands and GitHub push instructions are documented in README.md.")

print("\n" + "=" * 82)
print("TASK 8 - RENDER DEPLOYMENT")
print("=" * 82)
print("\nRecommended Render settings:")
print("Build Command:")
print("pip install -r requirements.txt")
print("\nStart Command:")
print("streamlit run app.py --server.address 0.0.0.0 --server.port $PORT")
print("\nThe included render.yaml contains the same deployment configuration.")

print("\n" + "=" * 82)
print("TASK 9 - FINAL VALIDATION")
print("=" * 82)
print("\nLocal recommendation logic was validated with three movie titles.")
print("The final public Render URL can only be added after the user's GitHub repository")
print("is connected to their Render account and the web service is deployed.")

print("\n" + "=" * 82)
print("ASSIGNMENT 20 COMPLETED SUCCESSFULLY")
print("=" * 82)
