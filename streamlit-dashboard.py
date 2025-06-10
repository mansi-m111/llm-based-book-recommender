import os
from dotenv import load_dotenv
from tqdm import tqdm

import streamlit as st

import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

persist_dir = "chroma_db"

# Only create the vector store if it doesn't already exist
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    # Load and split documents
    loader = TextLoader('tagged_description.txt', encoding="utf-8")
    raw_documents = loader.load()
    textsplitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")

    print("Splitting documents...")
    documents = textsplitter.split_documents(raw_documents)

    # Create and persist the vector store
    print("Creating vector store with embeddings...")

    db_books = Chroma.from_documents(documents,
                                     embedding=OpenAIEmbeddings(),
                                     persist_directory=persist_dir)

else:
    # Load the existing persistent database
    print("Loading existing vector store...")
    db_books = Chroma(persist_directory=persist_dir,
                      embedding_function=OpenAIEmbeddings())
    

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    print("inside retrieve_semantic_recommendation def")
    print("Performing vector similarity search...")
    recs = db_books.similarity_search(query, k = initial_top_k)

    print("Matching ISBNs with metadata...")
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    #filtering by category
    print("Filtering by category...")
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    #filtering by tone
    print("Filtering by tone...")
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    
    print("inside recommend_books def")
    print("Retrieving book recommendations...")
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # caption = f"{row['title']} by {authors_str}: {truncated_description}"

        title = row["title"]
        authors = authors_str
        caption = truncated_description
        results.append((row["large_thumbnail"], title, authors, caption))
    
    return results


# App layout
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommender")

# Sidebar or main panel inputs
st.markdown("### Describe a book you'd like to read")

query = st.text_input("Please enter a description of a book:",
                      placeholder="e.g., A story about forgiveness")

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

category = st.selectbox("Select a category:", options=categories, index=0)
tone = st.selectbox("Select an emotional tone:", options=tones, index=0)

# Submit and get recommendations
if st.button("Find Recommendations") and query.strip():
    with st.spinner("Processing recommendations..."):
        results = recommend_books(query, category, tone)
    st.success("Done!")

    st.markdown("## Recommendations")
    cols = st.columns(3)

    for idx, (img_url, title, author, caption) in enumerate(results):
        with cols[idx % 3]:
            st.image(img_url, use_container_width=True)
            st.markdown(title)
            st.markdown(f"By: {author}")
            st.markdown(caption)
