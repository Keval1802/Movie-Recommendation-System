from recommender import load_and_prepare_data, build_recommender

print("TASK 4 - COSINE SIMILARITY")

df = load_and_prepare_data("movies.csv")
bundle = build_recommender(df)

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
