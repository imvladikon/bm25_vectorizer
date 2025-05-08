import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bm25_vectorizer import BM25Vectorizer


class BM25SearchEngine:
    def __init__(self, documents, document_ids=None, transformer="bm25", k1=1.5, b=0.75):
        self.documents = documents
        self.document_ids = document_ids if document_ids is not None else list(range(len(documents)))

        # Create and fit the BM25 vectorizer
        self.vectorizer = BM25Vectorizer(transformer=transformer, k1=k1, b=b)
        self.vectorizer.fit(documents)

        # Transform all documents
        self.document_vectors = self.vectorizer.transform(documents)

        # Compute pairwise similarity matrix (for "similar documents" feature)
        self.similarity_matrix = cosine_similarity(self.document_vectors)
        np.fill_diagonal(self.similarity_matrix, 0)  # Exclude self-similarities

    def search(self, query, top_n=5):
        """Search for documents matching the query"""
        query_vector = self.vectorizer.transform([query])

        # Compute similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        # Get top N results
        top_indices = similarities.argsort()[::-1][:top_n]

        # Format results
        results = []
        for idx in top_indices:
            results.append({"id": self.document_ids[idx], "document": self.documents[idx], "score": similarities[idx]})

        return results

    def find_similar(self, doc_id, top_n=3):
        """Find documents similar to the given document ID"""
        # Get index of the document
        try:
            doc_idx = self.document_ids.index(doc_id)
        except ValueError:
            return []

        # Get similarities for this document
        similarities = self.similarity_matrix[doc_idx]

        # Get top N similar documents
        top_indices = similarities.argsort()[::-1][:top_n]

        # Format results
        results = []
        for idx in top_indices:
            results.append({"id": self.document_ids[idx], "document": self.documents[idx], "score": similarities[idx]})

        return results


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Python programming language for data science and machine learning",
        "Introduction to natural language processing with Python",
        "Web development using JavaScript and Node.js",
        "Data analysis and visualization with pandas and matplotlib",
        "Machine learning algorithms for text classification",
        "Deep learning frameworks: TensorFlow and PyTorch",
        "Statistical methods for data analysis",
        "Building recommendation systems with collaborative filtering",
    ]

    # Create document IDs (could be database IDs, filenames, etc.)
    doc_ids = [f"doc_{i}" for i in range(len(documents))]

    # Initialize search engine
    search_engine = BM25SearchEngine(documents, doc_ids, transformer="bm25l", k1=1.2, b=0.8)

    # Example 1: Search for documents
    query = "Statistical methods for data analysis"  # "python data analysis"
    results = search_engine.search(query, top_n=3)

    print(f"Search results for '{query}':")
    for i, result in enumerate(results):
        print(f"{i + 1}. [{result['id']}] (Score: {result['score']:.4f}): {result['document']}")

    print("\n" + "-" * 50 + "\n")

    # Example 2: Find similar documents
    doc_id = "doc_0"  # The Python programming language document
    similar_docs = search_engine.find_similar(doc_id, top_n=2)

    print(f"Documents similar to [{doc_id}]: {documents[0]}")
    for i, result in enumerate(similar_docs):
        print(f"{i + 1}. [{result['id']}] (Score: {result['score']:.4f}): {result['document']}")
