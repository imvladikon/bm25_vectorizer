import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bm25_vectorizer import BM25Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Python programming language basics",
    "Introduction to Python for data science",
    "Java programming fundamentals",
    "Data science with Python and pandas",
    "Web development with JavaScript",
    "Machine learning algorithms in Python",
    "Data analytics with pandas",
]

vectorizer = BM25Vectorizer(transformer='bm25plus', k1=1.2, b=0.8, delta=0.5)
vectorizer.fit(corpus)

tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectorizer.fit(corpus)

document_vectors = vectorizer.transform(corpus)
tfidf_vectors = tf_idf_vectorizer.transform(corpus)

similarity_matrix = cosine_similarity(document_vectors)
similarity_matrix_tfidf = cosine_similarity(tfidf_vectors)

# Set diagonal elements to 0 to exclude self-similarities
np.fill_diagonal(similarity_matrix, 0)
np.fill_diagonal(similarity_matrix_tfidf, 0)

# Find the most similar document pairs
n_pairs = 3  # Number of top pairs to find
most_similar_pairs = []

for _ in range(n_pairs):
    # Find indices of max similarity
    i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    similarity = similarity_matrix[i, j]

    # Store the pair and their similarity
    most_similar_pairs.append((i, j, similarity))

    # Set this pair's similarity to 0 to find the next pair
    similarity_matrix[i, j] = 0
    similarity_matrix[j, i] = 0  # Ensure symmetry

# Display the most similar document pairs
print("Top similar document pairs:")
print("-" * 50)
for i, j, sim in most_similar_pairs:
    print(f"Similarity: {sim:.4f}")
    print(f"Document 1: {corpus[i]}")
    print(f"Document 2: {corpus[j]}")
    print("-" * 30)


# Find the most similar document pairs using TF-IDF
most_similar_pairs_tfidf = []
for _ in range(n_pairs):
    # Find indices of max similarity
    i, j = np.unravel_index(np.argmax(similarity_matrix_tfidf), similarity_matrix_tfidf.shape)
    similarity = similarity_matrix_tfidf[i, j]

    # Store the pair and their similarity
    most_similar_pairs_tfidf.append((i, j, similarity))

    # Set this pair's similarity to 0 to find the next pair
    similarity_matrix_tfidf[i, j] = 0
    similarity_matrix_tfidf[j, i] = 0  # Ensure symmetry

# Display the most similar document pairs using TF-IDF
print("Top similar document pairs using TF-IDF:")
print("-" * 50)
for i, j, sim in most_similar_pairs_tfidf:
    print(f"Similarity: {sim:.4f}")
    print(f"Document 1: {corpus[i]}")
    print(f"Document 2: {corpus[j]}")
    print("-" * 30)