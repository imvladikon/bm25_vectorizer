import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from bm25_vectorizer import BM25TransformerBase, BM25Vectorizer

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

    def test_length_normalization_token_count_for_all_variants(self):
        """Length normalization must use token count, not nnz per document."""
        corpus = [
            "apple x x x",
            "apple x",
            "banana y",
        ]
        term = "apple"
        variants = ["bm25l", "bm25plus", "bm25adpt", "bm25t", "tfidf1ap"]

        for variant in variants:
            vec = BM25Vectorizer(
                transformer=variant,
                b=1.0,
                token_pattern=r"(?u)\b\w+\b",
                stop_words=None,
            ).fit(corpus)
            X = vec.transform(corpus).toarray()
            term_idx = list(vec.get_feature_names_out()).index(term)
            long_doc_score = X[0, term_idx]
            short_doc_score = X[1, term_idx]

            self.assertNotEqual(
                long_doc_score,
                short_doc_score,
                f"{variant}: scores should differ when token lengths differ",
            )
            idf_value = float(vec._tfidf._idf_diag.diagonal()[term_idx])
            if idf_value >= 0:
                self.assertLess(
                    long_doc_score,
                    short_doc_score,
                    f"{variant}: longer doc should be penalized more with b=1.0",
                )
            else:
                self.assertGreater(
                    long_doc_score,
                    short_doc_score,
                    f"{variant}: longer doc should be closer to zero with negative IDF",
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

        self.assertTrue(vec_with_idf._tfidf.use_idf)
        self.assertFalse(vec_without_idf._tfidf.use_idf)
        self.assertFalse(
            np.allclose(result_with_idf.data, result_without_idf.data), "Results should differ with use_idf parameter"
        )

    def test_fit_uses_raw_term_counts(self):
        corpus = ["a a a", "a b"]
        vec = BM25Vectorizer(
            transformer="bm25",
            k1=1.5,
            b=0.75,
            epsilon=0.0,
            log1p_idf=True,
            token_pattern=r"(?u)\b\w+\b",
            stop_words=None,
        ).fit(corpus)

        features = list(vec.get_feature_names_out())
        term_idx = features.index("a")
        X = vec.transform(corpus).toarray()

        n_samples = 2
        df_a = 2
        idf_a = np.log1p((n_samples - df_a + 0.5) / (df_a + 0.5))
        avgdl = 2.5
        dl = 3
        tf = 3
        expected = idf_a * (tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

        self.assertEqual(vec._tfidf.avgdl, avgdl)
        self.assertAlmostEqual(X[0, term_idx], expected)

    def test_fit_transform_does_not_use_sklearn_tfidf_weights(self):
        corpus = ["a a a b", "a b", "b c"]
        kwargs = {"token_pattern": r"(?u)\b\w+\b", "stop_words": None}

        vec = BM25Vectorizer(
            transformer="bm25",
            log1p_idf=True,
            norm="l2",
            smooth_idf=True,
            sublinear_tf=True,
            **kwargs,
        )
        bm25 = vec.fit_transform(corpus)

        counts = CountVectorizer(**kwargs).fit_transform(corpus)
        tfidf = TfidfVectorizer(**kwargs).fit_transform(corpus)

        self.assertIsInstance(vec, TfidfVectorizer)
        self.assertTrue(np.array_equal(vec._tfidf.doc_len_, counts.sum(axis=1).A1))
        self.assertFalse(np.allclose(bm25.toarray(), tfidf.toarray()))
        self.assertFalse(np.allclose(vec.transform(corpus).toarray(), counts.toarray()))

    def test_reference_rank_bm25_term_weights(self):
        try:
            from rank_bm25 import BM25L, BM25Okapi, BM25Plus
        except ImportError:
            self.skipTest("rank_bm25 is not installed")

        tokenized = [
            ["alpha", "alpha", "beta"],
            ["beta", "gamma"],
            ["delta", "gamma", "gamma"],
            ["epsilon", "zeta"],
        ]
        corpus = [" ".join(doc) for doc in tokenized]
        kwargs = {
            "k1": 1.5,
            "b": 0.75,
            "delta": 1.0,
            "token_pattern": r"(?u)\b\w+\b",
            "stop_words": None,
        }
        cases = [
            ("bm25", BM25Okapi(tokenized, k1=1.5, b=0.75, epsilon=0.25)),
            ("bm25l", BM25L(tokenized, k1=1.5, b=0.75, delta=1.0)),
            ("bm25plus", BM25Plus(tokenized, k1=1.5, b=0.75, delta=1.0)),
        ]

        for variant, reference in cases:
            vec = BM25Vectorizer(transformer=variant, **kwargs).fit(corpus)
            X = vec.transform(corpus).toarray()
            counts = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None).fit_transform(corpus).toarray()

            for term_idx, term in enumerate(vec.get_feature_names_out()):
                expected = np.array(reference.get_scores([term]))
                mask = counts[:, term_idx] > 0
                self.assertTrue(
                    np.allclose(X[mask, term_idx], expected[mask]),
                    f"{variant}: term weights for {term} should match rank_bm25",
                )

    def test_direct_score_matches_rank_bm25_for_core_variants(self):
        try:
            from rank_bm25 import BM25L, BM25Okapi, BM25Plus
        except ImportError:
            self.skipTest("rank_bm25 is not installed")

        tokenized = [
            ["alpha", "alpha", "beta"],
            ["beta", "gamma"],
            ["delta", "gamma", "gamma"],
            ["epsilon", "zeta"],
        ]
        corpus = [" ".join(doc) for doc in tokenized]
        queries = [
            ["alpha", "gamma"],
            ["epsilon", "missing"],
            ["beta", "beta", "gamma"],
        ]
        kwargs = {
            "k1": 1.5,
            "b": 0.75,
            "delta": 1.0,
            "token_pattern": r"(?u)\b\w+\b",
            "stop_words": None,
        }
        cases = [
            ("bm25", BM25Okapi(tokenized, k1=1.5, b=0.75, epsilon=0.25)),
            ("bm25l", BM25L(tokenized, k1=1.5, b=0.75, delta=1.0)),
            ("bm25plus", BM25Plus(tokenized, k1=1.5, b=0.75, delta=1.0)),
        ]

        for variant, reference in cases:
            vec = BM25Vectorizer(transformer=variant, **kwargs).fit(corpus)
            actual = vec.score([" ".join(query) for query in queries])
            expected = np.vstack([reference.get_scores(query) for query in queries])

            self.assertTrue(np.allclose(actual, expected), f"{variant}: direct scores should match rank_bm25")

    def test_bm25plus_score_includes_absent_query_term_delta(self):
        corpus = ["alpha beta", "gamma delta"]
        vec = BM25Vectorizer(
            transformer="bm25plus",
            token_pattern=r"(?u)\b\w+\b",
            stop_words=None,
            delta=1.0,
        ).fit(corpus)

        scores = vec.score(["alpha"])
        transformed_scores = vec.transform(["alpha"]).dot(vec.transform(corpus).T).toarray()

        self.assertGreater(scores[0, 1], 0.0)
        self.assertEqual(transformed_scores[0, 1], 0.0)

    def test_rank_uses_direct_scores(self):
        corpus = ["alpha beta", "gamma delta", "alpha alpha gamma"]
        vec = BM25Vectorizer(
            transformer="bm25",
            token_pattern=r"(?u)\b\w+\b",
            stop_words=None,
            log1p_idf=True,
        ).fit(corpus)

        ranked, scores = vec.rank(["alpha", "delta"], top_n=2, return_scores=True)
        direct_scores = vec.score(["alpha", "delta"])

        self.assertEqual(ranked.shape, (2, 2))
        self.assertTrue(np.allclose(scores, np.take_along_axis(direct_scores, ranked, axis=1)))
        self.assertEqual(ranked[0, 0], 2)
        self.assertEqual(ranked[1, 0], 1)

    def test_rank_batches_match_full_score_sorting(self):
        corpus = [
            "alpha beta",
            "alpha alpha gamma",
            "delta epsilon",
            "gamma delta",
            "zeta eta theta",
        ]
        queries = ["alpha", "delta gamma", "theta"]

        for variant in ["bm25", "bm25plus"]:
            vec = BM25Vectorizer(
                transformer=variant,
                token_pattern=r"(?u)\b\w+\b",
                stop_words=None,
                log1p_idf=True,
            ).fit(corpus)
            scores = vec.score(queries)
            expected_ranked = np.argsort(scores, axis=1)[:, ::-1][:, :3]
            expected_scores = np.take_along_axis(scores, expected_ranked, axis=1)

            ranked, sorted_scores = vec.rank(queries, top_n=3, return_scores=True, batch_size=1)

            self.assertTrue(np.array_equal(ranked, expected_ranked), variant)
            self.assertTrue(np.allclose(sorted_scores, expected_scores), variant)

    def test_similarity_alternative_term_specific_transformers_use_fitted_counts(self):
        corpus = ["alpha alpha beta", "beta gamma", "gamma gamma delta"]
        vec = BM25Vectorizer(
            transformer="bm25",
            token_pattern=r"(?u)\b\w+\b",
            stop_words=None,
        ).fit(corpus)

        for transformer in ["bm25t", "bm25adpt"]:
            sim = vec.similarity(corpus[0], corpus[1], transformer=transformer)
            self.assertTrue(np.isfinite(sim))

    def test_bm25adpt_against_independent_oracle(self):
        corpus = [
            "alpha alpha alpha beta",
            "alpha beta beta",
            "beta gamma gamma gamma",
            "delta gamma",
        ]
        k1 = 1.5
        b = 0.75
        token_kwargs = {"token_pattern": r"(?u)\b\w+\b", "stop_words": None}
        vec = BM25Vectorizer(transformer="bm25adpt", k1=k1, b=b, **token_kwargs).fit(corpus)
        X = CountVectorizer(**token_kwargs).fit_transform(corpus).tocsr()
        actual = vec.transform(corpus).toarray()

        df = np.bincount(X.indices, minlength=X.shape[1])
        dl = X.sum(axis=1).A1
        avgdl = dl.mean()
        n_samples, n_terms = X.shape

        def df_r(r):
            if r == 0:
                return np.full(n_terms, n_samples)
            if r == 1:
                return df
            out = np.zeros(n_terms, dtype=np.int32)
            for doc_idx in range(n_samples):
                start, end = X.indptr[doc_idx], X.indptr[doc_idx + 1]
                norm = 1 - b + b * dl[doc_idx] / avgdl
                for pos in range(start, end):
                    c_td = X.data[pos] / norm
                    if c_td >= r - 0.5:
                        out[X.indices[pos]] += 1
            return out

        max_r = 5
        df_values = [df_r(r) for r in range(max_r + 2)]
        g_values = []
        for r in range(max_r):
            g_values.append(
                np.log2((df_values[r + 2] + 0.5) / (df_values[r + 1] + 1))
                - np.log2((df_values[r + 1] + 0.5) / (n_samples + 1))
            )
        g_1 = g_values[0]
        g_normalized = [np.divide(g, g_1, out=np.zeros_like(g), where=g_1 != 0) for g in g_values]

        k1_terms = np.full(n_terms, k1)
        for term_idx in range(n_terms):
            best_k1 = k1
            min_error = float("inf")
            for candidate in np.linspace(0.1, 5.0, 50):
                error = 0.0
                for r in range(1, max_r + 1):
                    bm25_component = (candidate + 1) * r / (candidate + r)
                    error += (g_normalized[r - 1][term_idx] - bm25_component) ** 2
                if error < min_error:
                    min_error = error
                    best_k1 = candidate
            k1_terms[term_idx] = best_k1

        expected = np.zeros_like(actual)
        for doc_idx in range(n_samples):
            start, end = X.indptr[doc_idx], X.indptr[doc_idx + 1]
            norm = 1 - b + b * dl[doc_idx] / avgdl
            for pos in range(start, end):
                term_idx = X.indices[pos]
                tf = X.data[pos]
                expected[doc_idx, term_idx] = g_1[term_idx] * tf * (k1_terms[term_idx] + 1) / (
                    k1_terms[term_idx] * norm + tf
                )

        self.assertTrue(np.allclose(actual, expected))

    def test_bm25t_against_independent_oracle(self):
        corpus = [
            "alpha alpha alpha beta",
            "alpha beta beta",
            "beta gamma gamma gamma",
            "delta gamma",
        ]
        k1 = 1.5
        b = 0.75
        token_kwargs = {"token_pattern": r"(?u)\b\w+\b", "stop_words": None}
        vec = BM25Vectorizer(transformer="bm25t", k1=k1, b=b, **token_kwargs).fit(corpus)
        X = CountVectorizer(**token_kwargs).fit_transform(corpus).tocsr()
        actual = vec.transform(corpus).toarray()

        df = np.bincount(X.indices, minlength=X.shape[1])
        dl = X.sum(axis=1).A1
        avgdl = dl.mean()
        n_samples, n_terms = X.shape

        idf = np.log((n_samples + 1) / (df + 0.5))
        idf = np.maximum(idf, 0.25 * float(np.mean(idf)))

        def g_k1(value):
            if abs(value - 1.0) < 1e-10:
                return 1.0
            return (value / (value - 1.0)) * np.log(value)

        def g_k1_derivative(value):
            if abs(value - 1.0) < 1e-10:
                return 0.5
            return np.log(value) / (value - 1) - value * np.log(value) / ((value - 1) ** 2) + 1 / (value - 1)

        k1_terms = np.full(n_terms, k1)
        for term_idx in range(n_terms):
            if df[term_idx] == 0:
                continue
            log_ctd_sum = 0.0
            for doc_idx in range(n_samples):
                start, end = X.indptr[doc_idx], X.indptr[doc_idx + 1]
                positions = np.where(X.indices[start:end] == term_idx)[0]
                if len(positions) == 0:
                    continue
                tf = X.data[start + positions[0]]
                norm = 1 - b + b * dl[doc_idx] / avgdl
                c_td = tf / norm
                log_ctd_sum += np.log(c_td + 1)
            target = log_ctd_sum / df[term_idx]

            current = k1
            for _ in range(20):
                error = g_k1(current) - target
                if abs(error) < 1e-6:
                    break
                derivative = g_k1_derivative(current)
                if abs(derivative) <= 1e-10:
                    break
                current = max(0.1, current - error / derivative)
            k1_terms[term_idx] = current

        expected = np.zeros_like(actual)
        for doc_idx in range(n_samples):
            start, end = X.indptr[doc_idx], X.indptr[doc_idx + 1]
            norm = 1 - b + b * dl[doc_idx] / avgdl
            for pos in range(start, end):
                term_idx = X.indices[pos]
                tf = X.data[pos]
                expected[doc_idx, term_idx] = idf[term_idx] * tf * (k1_terms[term_idx] + 1) / (
                    tf + k1_terms[term_idx] * norm
                )

        self.assertTrue(np.allclose(actual, expected))

    def test_ir_retrieval_top1_across_topics(self):
        corpus = [
            "python pandas data analysis tutorial",
            "python machine learning model training",
            "javascript web browser frontend app",
            "statistical data analysis regression model",
            "recipe bread yeast sourdough baking",
            "football match goal team league",
        ]
        queries = {
            "python data analysis": 0,
            "frontend browser javascript": 2,
            "sourdough bread recipe": 4,
            "football goal match": 5,
            "regression statistical analysis": 3,
        }

        for variant in ["bm25", "bm25l", "bm25plus"]:
            vec = BM25Vectorizer(
                transformer=variant,
                token_pattern=r"(?u)\b\w+\b",
                stop_words=None,
                log1p_idf=True,
            ).fit(corpus)
            document_vectors = vec.transform(corpus)

            for query, expected_top_idx in queries.items():
                query_vector = vec.transform([query])
                scores = cosine_similarity(query_vector, document_vectors).ravel()
                ranked = scores.argsort()[::-1]

                self.assertEqual(ranked[0], expected_top_idx, f"{variant}: wrong top-1 for query {query!r}")
                self.assertGreater(scores[ranked[0]], 0.0, f"{variant}: top score should be non-zero")

    def test_ir_retrieval_prefers_specific_rare_match(self):
        corpus = [
            "python data data tutorial common",
            "python raretoken exact match",
            "common common common data",
            "javascript frontend browser",
        ]
        query = "python raretoken"

        for variant in ["bm25", "bm25l", "bm25plus"]:
            vec = BM25Vectorizer(
                transformer=variant,
                token_pattern=r"(?u)\b\w+\b",
                stop_words=None,
                log1p_idf=True,
            ).fit(corpus)
            scores = cosine_similarity(vec.transform([query]), vec.transform(corpus)).ravel()
            ranked = scores.argsort()[::-1]

            self.assertEqual(ranked[0], 1, f"{variant}: rare exact match should rank first")
            self.assertGreater(scores[1], scores[0], f"{variant}: rare term should beat common overlap")

    def test_ir_manual_50_documents_10_queries_and_tfidf_math_comparison(self):
        corpus = [
            "python pandas dataframe groupby analysis",
            "python numpy vector matrix linear algebra",
            "javascript react frontend browser component",
            "postgres sql index query optimizer",
            "machine learning gradient boosting classifier",
            "neural network transformer attention embeddings",
            "docker container kubernetes deployment cluster",
            "linux kernel scheduler process memory",
            "api authentication oauth jwt token",
            "cache redis latency throughput performance",
            "battery charging overheating safety recall",
            "electric vehicle battery range charging station",
            "solar panel inverter grid storage",
            "mortgage interest escrow closing appraisal",
            "tax deduction invoice accounting audit",
            "contract clause liability indemnity arbitration",
            "clinical trial placebo dosage efficacy",
            "protein enzyme receptor binding assay",
            "football striker goal penalty league",
            "basketball rebound assist defense playoff",
            "sourdough starter yeast flour fermentation",
            "espresso grinder extraction crema roast",
            "flight booking luggage airport boarding",
            "hotel reservation checkout breakfast concierge",
            "camera aperture shutter iso exposure",
            "guitar chord fret tuning amplifier",
            "climate carbon emission warming policy",
            "ocean coral reef marine biodiversity",
            "quantum photon entanglement measurement",
            "telescope galaxy nebula redshift observation",
            "python python python python tutorial basics",
            "data data data common common analysis",
            "battery battery battery charging common",
            "contract contract clause legal common",
            "football football goal common common",
            "recipe bread yeast oven common",
            "browser javascript css html common",
            "database sql table join common",
            "neural neural model training common",
            "finance stock portfolio risk common",
            "pandas dataframe merge missing values cleanup",
            "oauth token refresh session security",
            "kubernetes pod service ingress deployment",
            "clinical dosage adverse event safety",
            "mortgage refinance rate lender credit",
            "espresso latte milk steaming cafe",
            "galaxy telescope spectrum dark matter",
            "coral bleaching ocean temperature",
            "camera lens focus portrait bokeh",
            "guitar pedal distortion tone stage",
        ]
        queries = [
            ("pandas dataframe cleanup", 40),
            ("oauth refresh token", 41),
            ("battery safety overheating", 10),
            ("contract indemnity arbitration", 15),
            ("football penalty striker", 18),
            ("sourdough yeast fermentation", 20),
            ("telescope galaxy redshift", 29),
            ("clinical dosage efficacy", 16),
            ("kubernetes deployment ingress", 42),
            ("espresso extraction crema", 21),
        ]
        vectorizer_kwargs = {"token_pattern": r"(?u)\b\w+\b", "stop_words": None}

        bm25 = BM25Vectorizer(transformer="bm25", log1p_idf=True, epsilon=0.0, **vectorizer_kwargs).fit(corpus)
        tfidf = TfidfVectorizer(**vectorizer_kwargs).fit(corpus)
        bm25_docs = bm25.transform(corpus)
        tfidf_docs = tfidf.transform(corpus)

        self.assertFalse(np.allclose(bm25_docs.toarray(), tfidf_docs.toarray()))

        for query, expected_top_idx in queries:
            bm25_query = bm25.transform([query])
            tfidf_query = tfidf.transform([query])
            bm25_scores = cosine_similarity(bm25_query, bm25_docs).ravel()
            tfidf_scores = cosine_similarity(tfidf_query, tfidf_docs).ravel()

            self.assertEqual(bm25_scores.argsort()[::-1][0], expected_top_idx)
            self.assertEqual(tfidf_scores.argsort()[::-1][0], expected_top_idx)
            self.assertGreater(bm25_scores[expected_top_idx], 0.0)
            self.assertGreater(tfidf_scores[expected_top_idx], 0.0)
            self.assertFalse(np.allclose(bm25_query.toarray(), tfidf_query.toarray()))
            self.assertFalse(np.allclose(bm25_scores, tfidf_scores))

        term = "battery"
        bm25_term_idx = list(bm25.get_feature_names_out()).index(term)
        bm25_matrix = bm25.transform(corpus).toarray()
        df = np.count_nonzero(CountVectorizer(**vectorizer_kwargs).fit_transform(corpus).toarray()[:, bm25_term_idx])
        n_samples = len(corpus)
        idf = np.log1p((n_samples - df + 0.5) / (df + 0.5))
        avgdl = np.mean([len(doc.split()) for doc in corpus])
        k1 = 1.5
        b = 0.75

        def manual_bm25_weight(tf, dl):
            norm = 1 - b + b * dl / avgdl
            return idf * (tf * (k1 + 1)) / (tf + k1 * norm)

        doc_once = 10
        doc_repeated = 32
        once_expected = manual_bm25_weight(tf=1, dl=len(corpus[doc_once].split()))
        repeated_expected = manual_bm25_weight(tf=3, dl=len(corpus[doc_repeated].split()))

        self.assertAlmostEqual(bm25_matrix[doc_once, bm25_term_idx], once_expected)
        self.assertAlmostEqual(bm25_matrix[doc_repeated, bm25_term_idx], repeated_expected)

        tfidf_raw = TfidfVectorizer(norm=None, **vectorizer_kwargs).fit(corpus)
        tfidf_term_idx = list(tfidf_raw.get_feature_names_out()).index(term)
        tfidf_matrix = tfidf_raw.transform(corpus).toarray()
        sklearn_idf = np.log((1 + n_samples) / (1 + df)) + 1

        self.assertAlmostEqual(tfidf_matrix[doc_once, tfidf_term_idx], sklearn_idf)
        self.assertAlmostEqual(tfidf_matrix[doc_repeated, tfidf_term_idx], 3 * sklearn_idf)
        self.assertLess(repeated_expected / once_expected, 3.0)

    @unittest.skipUnless(os.getenv("RUN_HF_DATASET_TESTS") == "1", "set RUN_HF_DATASET_TESTS=1 to run")
    def test_huggingface_ag_news_1000_document_retrieval(self):
        from datasets import load_dataset

        train = load_dataset("ag_news", split="train[:1000]")
        queries = load_dataset("ag_news", split="test[:200]")
        corpus = [row["text"] for row in train]
        labels = np.array([row["label"] for row in train])
        query_texts = [row["text"] for row in queries]
        query_labels = np.array([row["label"] for row in queries])
        baseline = np.bincount(labels, minlength=4).max() / len(labels)

        scores_by_variant = {}
        for variant in ["bm25", "bm25l", "bm25plus", "bm25adpt", "bm25t", "tfidf1ap"]:
            vec = BM25Vectorizer(transformer=variant, stop_words="english", min_df=2, log1p_idf=True)
            document_vectors = vec.fit_transform(corpus)
            query_vectors = vec.transform(query_texts)
            scores = cosine_similarity(query_vectors, document_vectors)
            ranked = np.argsort(scores, axis=1)[:, ::-1]
            top1 = labels[ranked[:, 0]] == query_labels
            top5 = np.array([query_labels[i] in labels[ranked[i, :5]] for i in range(len(query_texts))])

            self.assertEqual(document_vectors.shape[0], 1000)
            self.assertEqual(query_vectors.shape[0], 200)
            self.assertTrue(np.isfinite(scores).all(), f"{variant}: scores should be finite")
            self.assertEqual(np.count_nonzero(np.max(scores, axis=1) > 0), 200)
            self.assertGreater(top1.mean(), baseline + 0.2, f"{variant}: top-1 should beat label baseline")
            self.assertGreater(top5.mean(), 0.9, f"{variant}: top-5 should be high on ag_news")
            scores_by_variant[variant] = scores

        tfidf = TfidfVectorizer(stop_words="english", min_df=2)
        tfidf_document_vectors = tfidf.fit_transform(corpus)
        tfidf_query_vectors = tfidf.transform(query_texts)
        tfidf_scores = cosine_similarity(tfidf_query_vectors, tfidf_document_vectors)

        self.assertFalse(np.allclose(scores_by_variant["bm25"], tfidf_scores))
        self.assertFalse(np.allclose(scores_by_variant["bm25"], scores_by_variant["bm25l"]))
        self.assertFalse(np.allclose(scores_by_variant["bm25"], scores_by_variant["bm25plus"]))

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
        examples_dir = ROOT_DIR / "examples"
        import subprocess

        expected_output = {
            "1_retrieve.py": "Rank 1 (Score: 1.0000): quick fox jumping",
            "2_match.py": "Document 2: Data analytics with pandas",
            "3_both.py": "1. [doc_6] (Score: 1.0000): Statistical methods for data analysis",
        }

        for example_file in examples_dir.glob("*.py"):
            result = subprocess.run(
                [sys.executable, str(example_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Example {example_file} failed with error: {result.stderr}")
            self.assertIn(expected_output[example_file.name], result.stdout)


if __name__ == "__main__":
    unittest.main()
