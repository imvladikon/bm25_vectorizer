import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bm25_vectorizer import BM25Vectorizer

# Sample corpus of documents
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A fast fox quickly leaps above a sleeping canine",
    "quick fox jumping",
    "The lazy dog sleeps all day",
    "Document retrieval and information retrieval systems",
    "quick something fox something jumps",
    "Machine learning for text ranking and retrieval",
    "Natural language processing techniques",
]

vectorizer = BM25Vectorizer(transformer="bm25plus", k1=1.5, b=0.75)
vectorizer.fit(corpus)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)

# Transform the corpus into BM25 vectors
document_vectors = vectorizer.transform(corpus)
# Transform the corpus into TF-IDF vectors
tfidf_vectors = tfidf_vectorizer.transform(corpus)

# Create a search query and transform it
query = "quick fox jumping"
query_vector = vectorizer.transform([query])
# Transform the query into TF-IDF vector
query_tfidf_vector = tfidf_vectorizer.transform([query])

# Calculate cosine similarity between query and all documents
similarities = cosine_similarity(query_vector, document_vectors).flatten()
# Calculate cosine similarity between query and all documents using TF-IDF
tfidf_similarities = cosine_similarity(query_tfidf_vector, tfidf_vectors).flatten()

# Rank documents by similarity score
ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order
ranked_tfidf_indices = np.argsort(tfidf_similarities)[::-1]  # Sort in descending order

# Display ranked results
print("Search results for query:", query)
print("-" * 50)
for i, idx in enumerate(ranked_indices):
    print(f"Rank {i + 1} (Score: {similarities[idx]:.4f}): {corpus[idx]}")

print("-" * 50)
print("Search results for query using TF-IDF:", query)
print("-" * 50)
for i, idx in enumerate(ranked_tfidf_indices):
    print(f"Rank {i + 1} (Score: {tfidf_similarities[idx]:.4f}): {corpus[idx]}")
