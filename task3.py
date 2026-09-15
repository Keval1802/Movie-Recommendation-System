import sys
sys.stdout.reconfigure(encoding="utf-8")
from recommender import load_and_prepare_data, build_recommender

print("TASK 3 - TF-IDF VECTORIZATION")

df = load_and_prepare_data("movies.csv")
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
