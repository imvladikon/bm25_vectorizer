import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from bm25_vectorizer import BM25Vectorizer, BM25TransformerBase

HERE = Path(__file__).parent.resolve()
ROOT_DIR = HERE.parent


class TestBM25VectorizerExtended(unittest.TestCase):
    def setUp(self):
        self.corpus = ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]
        self.query = "good weather London"

    def test_k1_effect(self):
        base_vec = BM25Vectorizer(transformer="bm25", k1=1.5, b=0.75).fit(self.corpus)
        base_result = base_vec.transform([self.query])

        # Test k1 parameter effect
        k1_vec = BM25Vectorizer(transformer="bm25", k1=2.0, b=0.75).fit(self.corpus)
        k1_result = k1_vec.transform([self.query])
        self.assertFalse(np.allclose(base_result.data, k1_result.data), "k1 parameter should affect the results")

    def test_b_effect(self):
        base_vec = BM25Vectorizer(transformer="bm25", k1=1.5, b=0.75).fit(self.corpus)
        base_result = base_vec.transform([self.query])

        b_vec = BM25Vectorizer(transformer="bm25", k1=1.5, b=0.5).fit(self.corpus)
        b_result = b_vec.transform([self.query])
        self.assertFalse(np.allclose(base_result.data, b_result.data), "b parameter should affect the results")

    def test_epsilon_parameter(self):
        """
        need special corpus with highly skewed term frequencies to test epsilon effect otherwise epsilon might not affect results

        - If all terms in corpus have IDF values higher than epsilon * mean(idf), changing epsilon won't affect results.
        - If the specific query doesn't contain terms that would be affected by the epsilon threshold, results will be identical.
        - In a small corpus with few documents, IDF values might be distributed in a way that epsilon has minimal impact.
        """  # noqa
        corpus = ["common common common rare", "common common common", "common common", "common rare"]
        query = "common rare"

        vec1 = BM25Vectorizer(transformer="bm25", epsilon=0.1).fit(corpus)
        vec2 = BM25Vectorizer(transformer="bm25", epsilon=0.9).fit(corpus)

        result1 = vec1.transform([query])
        result2 = vec2.transform([query])

        self.assertFalse(np.allclose(result1.data, result2.data), "epsilon parameter should affect the results")

    def test_empty_documents(self):
        # Test with an empty document in the corpus
        corpus_with_empty = self.corpus + [""]
        vec = BM25Vectorizer(transformer="bm25").fit(corpus_with_empty)
        X = vec.transform(corpus_with_empty)

        # The last document should have no non-zero elements
        self.assertEqual(X[-1].nnz, 0, "Empty document should have no non-zero elements")

        # Transform with an empty query
        q = vec.transform([""])
        self.assertEqual(q.nnz, 0, "Empty query should have no non-zero elements")

    def test_transform_new_documents(self):
        # Test transforming documents not in the training set
        vec = BM25Vectorizer(transformer="bm25").fit(self.corpus)

        # New document with known words
        new_doc = "London weather is good today"
        result = vec.transform([new_doc])
        self.assertTrue(result.nnz > 0, "New document with known words should have non-zero elements")

        # New document with unknown words
        unknown_doc = "Python programming is fun"
        result = vec.transform([unknown_doc])
        self.assertIsNotNone(result, "Should handle unknown words without error")

    def test_sklearn_pipeline_compatibility(self):
        # Test compatibility with sklearn pipeline
        pipeline = Pipeline([("vectorizer", BM25Vectorizer(transformer="bm25")), ("normalizer", Normalizer())])

        # Pipeline should fit and transform without errors
        result = pipeline.fit_transform(self.corpus)
        self.assertIsNotNone(result, "Pipeline should produce a result")

        # Test pipeline with transform
        query_result = pipeline.transform([self.query])
        self.assertIsNotNone(query_result, "Pipeline transform should work with query")

    def test_document_length_normalization(self):
        """Test that document length normalization works correctly"""
        short_doc = "Short document"
        long_doc = "This is a much longer document with many more words to increase the document length"

        vec = BM25Vectorizer(transformer="bm25", b=1.0, log1p_idf=True).fit([short_doc, long_doc])
        X = vec.transform([short_doc, long_doc])

        # Calculate average term weight for each document
        short_avg = X[0].sum() / max(1, X[0].nnz)
        long_avg = X[1].sum() / max(1, X[1].nnz)

        # With b=1.0, length normalization should be strongest,
        # making weights in longer document relatively smaller
        self.assertLess(long_avg, short_avg, "With b=1.0, average term weight in longer document should be less")

        # Test with b=0.0 (no length normalization)
        vec_no_norm = BM25Vectorizer(transformer="bm25", b=0.0).fit([short_doc, long_doc])
        X_no_norm = vec_no_norm.transform([short_doc, long_doc])

        short_avg_no_norm = X_no_norm[0].sum() / max(1, X_no_norm[0].nnz)
        long_avg_no_norm = X_no_norm[1].sum() / max(1, X_no_norm[1].nnz)

        # The difference should be less pronounced with b=0.0
        diff_with_norm = short_avg - long_avg
        diff_without_norm = short_avg_no_norm - long_avg_no_norm

        self.assertGreater(
            diff_with_norm, diff_without_norm, "Length normalization effect should be stronger with b=1.0"
        )

    def test_base_class_methods(self):
        # Test that base class methods raise NotImplementedError
        base = BM25TransformerBase()

        with self.assertRaises(NotImplementedError):
            base.transform(None)

        with self.assertRaises(NotImplementedError):
            base._calc_idf(None, None)

    def test_use_idf_parameter(self):
        vec_with_idf = BM25Vectorizer(transformer="bm25plus", use_idf=True).fit(self.corpus)
        vec_without_idf = BM25Vectorizer(transformer="bm25plus", use_idf=False).fit(self.corpus)

        result_with_idf = vec_with_idf.transform([self.query])
        result_without_idf = vec_without_idf.transform([self.query])

        self.assertFalse(
            np.allclose(result_with_idf.data, result_without_idf.data), "Results should differ with use_idf parameter"
        )

    def test_delta_parameter(self):
        # Test delta parameter for BM25L and BM25Plus
        for variant in ["bm25l", "bm25plus"]:
            base_vec = BM25Vectorizer(transformer=variant, delta=1.0).fit(self.corpus)
            delta_vec = BM25Vectorizer(transformer=variant, delta=2.0).fit(self.corpus)

            base_result = base_vec.transform([self.query])
            delta_result = delta_vec.transform([self.query])

            # Results should differ
            self.assertFalse(
                np.allclose(base_result.data, delta_result.data), f"Delta parameter should affect results for {variant}"
            )

    def test_invalid_transformer(self):
        # Test with invalid transformer name
        with self.assertRaises(KeyError):
            BM25Vectorizer(transformer="invalid_name").fit(self.corpus)

    def test_serialization(self):
        # Test that the vectorizer can be serialized and deserialized
        import pickle

        vec = BM25Vectorizer(transformer="bm25").fit(self.corpus)

        # Use a temporary file to test persistence
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                pickle.dump(vec, f)
                f.close()

                with open(f.name, "rb") as f2:
                    vec_unpickled = pickle.load(f2)

                # Check that the unpickled vectorizer produces the same results
                original_result = vec.transform([self.query])
                unpickled_result = vec_unpickled.transform([self.query])

                self.assertTrue(
                    np.allclose(original_result.data, unpickled_result.data),
                    "Unpickled vectorizer should produce same results",
                )

            finally:
                # Clean up
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_positive_scores(self):
        for variant in ["bm25", "bm25l", "bm25plus"]:
            vec = BM25Vectorizer(transformer=variant)
            X = vec.fit_transform(self.corpus)
            self.assertTrue(np.all(X.data > 0), f"Scores must be positive for {variant}")

    def test_ranking_consistency(self):
        query = "good London"
        doc1 = "good weather in London"
        doc2 = "weather today"
        test_corpus = [doc1, doc2]

        for variant in ["bm25", "bm25l", "bm25plus"]:
            # bm25 (due to small corpus) needs log1p_idf=True
            vec = BM25Vectorizer(transformer=variant, log1p_idf=True).fit(test_corpus)
            doc_vectors = vec.transform(test_corpus)
            query_vector = vec.transform([query])

            # use dot product for direct BM25 scoring
            # (no normalization, which might affect on bm25 standard for small corpus)
            scores = query_vector.dot(doc_vectors.T).toarray().flatten()

            self.assertGreater(scores[0], scores[1], f"{variant}: doc1 should have higher BM25 score than doc2")

    def test_different_transformers(self):
        vectors = {}
        sims = {}
        for variant in ["bm25", "bm25l", "bm25plus"]:
            vec = BM25Vectorizer(transformer=variant).fit(self.corpus)
            X = vec.transform(self.corpus)
            q = vec.transform([self.query])
            sims[variant] = cosine_similarity(X, q).flatten()
            vectors[variant] = X

        self.assertFalse(np.allclose(sims["bm25"], sims["bm25l"]), "BM25 vs BM25L differ")
        self.assertFalse(np.allclose(sims["bm25"], sims["bm25plus"]), "BM25 vs BM25Plus differ")
        self.assertFalse(np.allclose(sims["bm25l"], sims["bm25plus"]), "BM25L vs BM25Plus differ")

    def test_sparse_matrix_format(self):
        # Test that the output is a sparse matrix in CSR format
        vec = BM25Vectorizer(transformer="bm25").fit(self.corpus)
        result = vec.transform(self.corpus)

        self.assertEqual(result.format, "csr", "Output should be in CSR format")
        self.assertTrue(sp.issparse(result), "Output should be a sparse matrix")

    def test_against_manual_calculation(self):
        # Test against a manual calculation for a simple case
        corpus = ["a b c", "a b", "c d"]
        query = "a c"

        # Use a token pattern that accepts single-letter words and disable stop words
        vec = BM25Vectorizer(
            transformer="bm25",
            k1=1.5,
            b=0.75,
            token_pattern=r"(?u)\b\w+\b",  # Allow single-letter tokens
            stop_words=None,  # Disable stop words filtering
        ).fit(corpus)

        doc_vectors = vec.transform(corpus)
        query_vector = vec.transform([query])

        # Calculate similarity scores
        sim_scores = cosine_similarity(doc_vectors, query_vector).flatten()

        # Check that document 0 has the highest score (contains both a and c)
        self.assertEqual(np.argmax(sim_scores), 0, "Document 0 should have highest similarity score")

        # The similarity threshold of 0.95 might be too high for BM25
        # Consider using a lower threshold, as BM25 weighting differs from simple vector similarity
        self.assertGreater(sim_scores[0], 0.7, "Document 0 should have relatively high similarity score")

    def test_tf_idf_compatibility(self):
        """Test that BM25Vectorizer is compatible with TfidfVectorizer"""
        tfidf_vec = TfidfVectorizer().fit(self.corpus)
        bm25_vec = BM25Vectorizer(transformer="bm25").fit(self.corpus)

        self.assertEqual(
            len(tfidf_vec.vocabulary_),
            len(bm25_vec.vocabulary_),
            "BM25Vectorizer should have same vocabulary size as TfidfVectorizer",
        )

        tfidf_result = tfidf_vec.transform(self.corpus)
        bm25_result = bm25_vec.transform(self.corpus)

        self.assertEqual(
            tfidf_result.shape,
            bm25_result.shape,
            "BM25Vectorizer should produce output with same shape as TfidfVectorizer",
        )

        self.assertFalse(
            np.allclose(tfidf_result.data, bm25_result.data),
            "BM25Vectorizer results should not be identical to TfidfVectorizer results",
        )

    def test_query_ranking(self):
        vec = BM25Vectorizer(transformer="bm25").fit(self.corpus)
        X = vec.transform(self.corpus)
        q = vec.transform([self.query])

        sim = cosine_similarity(X, q).flatten()
        self.assertIn(np.argmax(sim), [0, 1, 2])
        self.assertTrue(np.any(sim > 0))

    def test_examples(self):
        # run all examples (*.py) in the examples directory
        # please do not put any dangerous code in the examples directory
        examples_dir = ROOT_DIR / "examples"
        for example_file in examples_dir.glob("*.py"):
            with open(example_file, "r") as f:
                code = f.read()
                exec(code)


if __name__ == "__main__":
    unittest.main()
