import sys
sys.stdout.reconfigure(encoding="utf-8")
from recommender import load_and_prepare_data, build_recommender, recommend

print("TASK 5 - RECOMMENDATION FUNCTION")

df = load_and_prepare_data("movies.csv")
bundle = build_recommender(df)

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
