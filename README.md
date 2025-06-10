# 📚 Intelligent Book Recommendation System with LLMs, Sentiment & Topic Filtering

This project delivers a flexible **Book Recommendation System** that leverages **Large Language Models (LLMs)** for semantic search, topic classification, and sentiment analysis. Users can discover books not only by description similarity but also by **category** and the **emotional tone** of the book, offering a highly personalized and nuanced search experience.

---

## 📌 Project Overview

**Objective:**  
Recommend books based on **semantic similarity**, **topic**, and **tone** using advanced NLP techniques.

**Dataset:**  
Kaggle "7K Books" — 6,810 unique books with title, author, category, description, ratings, etc.

---

## ✨ Key Features

- LLM-based embeddings for semantic search
- LLM-driven topic classification
- Sentiment analysis for tone detection
- Streamlit dashboard for interactive user experience
- Persistent ChromaDB vector store for efficient, reusable search

---

## 🎯 User Features

- 🔍 **Semantic Search**  
  Enter a phrase or description to find books with similar themes and narratives.

- 🗂️ **Category Filtering**  
  Select genres like *Fiction*, *Nonfiction*, or *Children's fiction* to narrow your search.

- 🎭 **Tone Filtering**  
  Choose an emotional tone (e.g., *happy*, *anger*, *Suspenseful*) based on sentiment analysis of book descriptions.

- 🚀 **Fast Results**  
  Embeddings are stored in a persistent ChromaDB vector database, ensuring rapid results and no repeated computation.

---

## 🧰 Tools & Technologies

**Languages:**  
- Python

**Libraries & Frameworks:**  
- `pandas` – Data handling  
- `transformers`, `openai`, `hugging-face models` – LLM embeddings and classification  
- `chromadb` – Vector storage and similarity search  
- `streamlit` – Dashboard UI

---

## 🧪 Methodology

### 1. **Data Preparation**
- Load and clean the Kaggle books dataset.
- Extract relevant fields: `title`, `author`, `category`, `description`.

### 2. **LLM Embedding Generation**
- Use `OpenAIEmbeddings` to create dense vectors for book descriptions.
- Store embeddings in **ChromaDB** for efficient semantic search.

### 3. **Topic Classification**
- Use **zero-shot classification** with `facebook/bart-large-mnli` to identify the main topic of each book.

### 4. **Sentiment Analysis**
- Analyze the emotional tone of each book using the model `j-hartmann/emotion-english-distilroberta-base`.

### 5. **Recommendation Engine**
- On user input, embed the query description.
- Filter books by selected category and tone.
- Use ChromaDB to find and return the most semantically similar results.

### 6. **Dashboard Interface**
- Developed with Streamlit.
- Enables search by text input, category dropdown, and tone filter.
- Displays top book recommendations with title, author, description and cover image.

---

## 📊 Key Highlights

- 🔎 **Personalized Discovery**: Find books by what they’re about, not just who wrote them.
- 🧠 **Nuanced Filtering**: Combine **meaning**, **topic**, and **emotion** for deeply tailored recommendations.
- ⚙️ **Efficient & Scalable**: ChromaDB avoids repeated computation and ensures responsiveness.
- 💡 **Modern NLP**: Powered by cutting-edge transformer models for both semantic and classification tasks.

---

## 🗃️ Models Used

| Task | Model |
|------|-------|
| Embeddings | `OpenAIEmbeddings` |
| Topic Classification | `facebook/bart-large-mnli` |
| Sentiment Analysis | `j-hartmann/emotion-english-distilroberta-base` |

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run streamlit-dashboard.py
