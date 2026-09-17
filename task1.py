from recommender import load_and_prepare_data

print("TASK 1 - LOAD & UNDERSTAND DATASET")

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
