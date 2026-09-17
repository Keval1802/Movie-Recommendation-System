from recommender import load_and_prepare_data

print("TASK 2 - TEXT PREPROCESSING")

df = load_and_prepare_data("movies.csv")

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
