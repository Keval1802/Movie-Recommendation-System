import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "movies.csv"


def clean_genres(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()

    if text.lower() == "(no genres listed)":
        return ""

    text = text.lower().replace("|", " ")
    text = re.sub(r"[^a-z\s-]", " ", text)

    tokens = [
        token
        for token in text.split()
        if token not in ENGLISH_STOP_WORDS
    ]

    return " ".join(tokens)


def load_and_prepare_data(csv_path=DATA_FILE):
    df = pd.read_csv(csv_path)
    df["genres"] = df["genres"].fillna("")
    df["clean_text"] = df["genres"].apply(clean_genres)
    return df


def build_recommender(df):
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

    unique_profiles = df["clean_text"].drop_duplicates().tolist()
    profile_to_id = {
        profile: index for index, profile in enumerate(unique_profiles)
    }

    movie_profile_ids = (
        df["clean_text"]
        .map(profile_to_id)
        .to_numpy(dtype=np.int32)
    )

    profile_tfidf = vectorizer.transform(unique_profiles)
    similarity_matrix = cosine_similarity(profile_tfidf, profile_tfidf)

    title_to_indices = {}
    for index, title in enumerate(df["title"].astype(str)):
        title_to_indices.setdefault(title, []).append(index)

    return {
        "df": df,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "unique_profiles": unique_profiles,
        "movie_profile_ids": movie_profile_ids,
        "similarity_matrix": similarity_matrix,
        "title_to_indices": title_to_indices,
    }


def recommend(item_name, bundle, top_n=5):
    df = bundle["df"]
    title_to_indices = bundle["title_to_indices"]

    if item_name not in title_to_indices:
        matches = df.index[
            df["title"].str.lower() == str(item_name).lower()
        ].tolist()

        if not matches:
            raise ValueError(f"Movie not found: {item_name}")

        selected_index = matches[0]
    else:
        selected_index = title_to_indices[item_name][0]

    selected_clean_text = df.at[selected_index, "clean_text"]

    if not selected_clean_text:
        return pd.DataFrame(
            columns=["title", "genres", "similarity"]
        )

    selected_profile_id = bundle["movie_profile_ids"][selected_index]
    profile_scores = bundle["similarity_matrix"][selected_profile_id]

    movie_scores = profile_scores[bundle["movie_profile_ids"]].copy()
    movie_scores[selected_index] = -1.0

    n_items = len(movie_scores)
    top_n = max(1, min(int(top_n), n_items - 1))

    candidate_count = min(n_items, max(top_n * 20, 100))
    candidate_indices = np.argpartition(
        movie_scores,
        -candidate_count
    )[-candidate_count:]

    ordered_indices = candidate_indices[
        np.argsort(movie_scores[candidate_indices])[::-1]
    ]

    results = []
    seen_titles = {str(df.at[selected_index, "title"])}

    for idx in ordered_indices:
        title = str(df.at[idx, "title"])

        if title in seen_titles:
            continue

        seen_titles.add(title)

        results.append({
            "title": title,
            "genres": str(df.at[idx, "genres"]),
            "similarity": float(movie_scores[idx]),
        })

        if len(results) == top_n:
            break

    return pd.DataFrame(results)
