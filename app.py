import streamlit as st

from recommender import load_and_prepare_data, build_recommender, recommend


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource(show_spinner="Building recommendation engine...")
def get_recommender():
    df = load_and_prepare_data()
    bundle = build_recommender(df)
    return bundle


bundle = get_recommender()
df = bundle["df"]

st.title("🎬 Movie Recommendation System")
st.write(
    "Content-based recommendations using MovieLens genres, "
    "TF-IDF and cosine similarity."
)

st.caption(
    f"Dataset: MovieLens Latest Full • {len(df):,} movies"
)

movie_titles = df["title"].astype(str).drop_duplicates().tolist()

selected_movie = st.selectbox(
    "Select a movie",
    movie_titles,
    index=movie_titles.index("Toy Story (1995)")
    if "Toy Story (1995)" in movie_titles
    else 0,
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5,
)

if st.button("Get Recommendations", type="primary"):
    results = recommend(
        selected_movie,
        bundle,
        top_n=top_n,
    )

    st.subheader(f"Movies similar to {selected_movie}")

    if results.empty:
        st.warning(
            "This movie has no listed genres in the dataset, "
            "so genre-based recommendations cannot be generated."
        )
    else:
        display = results.copy()
        display["similarity"] = (
            display["similarity"] * 100
        ).round(2).astype(str) + "%"

        display = display.rename(
            columns={
                "title": "Movie",
                "genres": "Genres",
                "similarity": "Similarity",
            }
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Method: cleaned genres → TF-IDF (1–2 grams) → cosine similarity."
)
