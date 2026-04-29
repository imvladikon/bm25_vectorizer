from typing import Any, Iterable, Literal, Optional, Type

import numpy as np
import scipy.sparse as sp
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self, override

"""
BM25 implementation for scikit-learn.

This module provides BM25, BM25L, and BM25Plus transformers and vectorizers
compatible with scikit-learn's API. 
They can be used as alternatives to TF-IDF for text ranking and retrieval tasks.
"""  # noqa

__version__ = "0.0.4"


class BM25TransformerBase(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        log1p_idf: bool = False,
        use_idf: bool = True,
    ) -> None:
        """
        k1 : float, default=1.5
             controls term frequency saturation, helps with:
             - how much additional occurrences of a term in a document contribute to the relevance score
               higher values give more weight to term frequency.
             - when there are documents with varying term frequencies
             - when terms appear with different frequencies, controls how much these repeated occurrences matter
             - short vs long documents with repeated terms (how much those repetitions contribute to relevance)

        b : float, default=0.75
             controls document length normalization how much the relevance scoring should be adjusted
                   based on a document's length relative to the average document length in the collection
             - 0 means no length normalization
             - 1 means full normalization.
             higher values give more penalty to longer documents.

        delta : float, default=1.0
             depends on the variant of BM25, general purpose as a "boosting" or "smoothing" parameter
             that helps ensure terms make meaningful contributions to relevance scores
             regardless of document length or frequency characteristics.

        epsilon: lower-bounding of IDF values, how very common terms are scored in the relevance ranking:
           - higher (0.5-1.0): common terms retain more influence in scoring, reducing the gap between rare and common terms
           - lower  (0.1-0.2): common terms have minimal impact, creating greater distinction between rare and common terms
           - 0: common terms could have negative IDF, potentially reducing document scores when they contain extremely common terms

        log1p_idf: bool = False
            whether to use log1p (log(1 + x)) - smoothed Robertson–Sparck Jones IDF
            instead of log (log(x)) for IDF calculation to smooth out the IDF values
            and prevent extreme values from dominating the scoring.

            Setting it to True guarantees non-negative IDF values even in
            very small corpora (N ≤ 2), at the cost of slightly compressing
            the range of rare-term weights.

        use_idf: whether to include the inverse document frequency (IDF) component in the scoring.
            If True, rare terms will have higher weight than common terms.
            If False, term rarity is ignored and only term frequency and document
            length normalization are considered.

            Setting use_idf=False effectively transforms the ranking function into
            a pure term frequency model without considering document frequency.
            This might be useful when working with a very small corpus or in special
            cases where term rarity should be ignored.
        """  # noqa
        self.k1: float = k1
        self.b: float = b
        self.delta: float = delta
        self.epsilon: float = epsilon
        self.use_idf: bool = use_idf
        self._idf_diag: Optional[csr_matrix] = None
        self.avgdl: float | None = None
        self.log1p_idf: bool = log1p_idf

    def fit(self, X: csr_matrix) -> Self:
        """
        Fit the transformer on the input sparse matrix.
        Calculates IDF and average document length.
        """  # noqa
        df: ndarray = np.bincount(X.indices, minlength=X.shape[1])
        return self.fit_from_stats(df, X.shape[0], X.sum(axis=1).A1)

    def fit_from_stats(self, df: ndarray, n_samples: int, doc_len: ndarray) -> Self:
        idf: ndarray = self._calc_idf(df, n_samples)
        self._idf_diag = sp.diags(idf, offsets=0, format="csr")
        self.doc_len_ = doc_len
        self.avgdl = self.doc_len_.mean()
        return self

    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        raise NotImplementedError

    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        raise NotImplementedError


class BM25Transformer(BM25TransformerBase):
    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """
        Compute inverse-document frequency (IDF) for BM25(Robertson-Sparck Jones IDF formula):

                     N - n_t + 0.5
        idf(t) = ln(--------------)
                     n_t + 0.5

        where
          N   = total number of documents
          n_t = number of documents that contain term t

        prevents negative IDF values when term frequency exceeds half the collection size.
        """  # noqa
        if self.log1p_idf:
            idf: ndarray = np.log1p((n_samples - df + 0.5) / (df + 0.5))
        else:
            idf: ndarray = np.log((n_samples - df + 0.5) / (df + 0.5))
        return np.where(idf < 0, self.epsilon * float(np.mean(idf)), idf)

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        BM25 score for term t in document d:

                                  tf_td · (k1 + 1)
        BM25(t, d) = idf(t) · ---------------------------
                            tf_td + k1 · (1 – b + b · |d|/avgdl)

        where
          tf_td  = term-frequency of t in d
          |d|   = document length
          avgdl = average document length in the corpus
          k1, b = BM25 parameters

        applying term frequency saturation and document length normalization.
        """  # noqa
        if copy:
            X = X.copy()
        dl = X.sum(axis=1).A1  # token length of each row in X
        rep: ndarray = np.repeat(dl, np.diff(X.indptr))
        data: ndarray = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / self.avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X = X @ self._idf_diag
        return X


class BM25LTransformer(BM25TransformerBase):
    """
    BM25L variant compatible with `rank_bm25`.

    Carries an extra ``tf_td`` multiplier outside the normalised-tf factor.
    This matches `rank_bm25.BM25L` exactly but deviates from the canonical
    Lv & Zhai (2011) / Kamphuis et al. (2020) formulation; for that, see
    :class:`BM25LCanonicalTransformer`.
    """  # noqa

    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """
        BM25L inverse-document frequency (IDF):

                       N + 1
        idf(t) = ln(-----------)
                     n_t + 0.5

        where
          N   = total number of documents
          n_t = number of documents that contain term t
        """  # noqa
        idf: ndarray = np.log((n_samples + 1) / (df + 0.5))
        return np.maximum(idf, self.epsilon * float(np.mean(idf)))

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        BM25L score (rank_bm25-compatible) for term t in document d:
                                  tf_td · (k1 + 1) · (c_td + δ)
        BM25L(t, d) = idf(t) · -------------------------------------
                                           k1 + c_td + δ
        with
          c_td = tf_td / (1 – b + b · |d| / avgdl)

        Note: the leading ``tf_td`` factor is the rank_bm25 form. The canonical
        Lv & Zhai (2011) BM25L (also implemented by `bm25s`) drops this factor;
        see :class:`BM25LCanonicalTransformer`.
        """  # noqa
        if copy:
            X = X.copy()
        nnz_per_doc: ndarray = np.diff(X.indptr)
        dl: ndarray = X.sum(axis=1).A1
        rep: ndarray = np.repeat(dl, nnz_per_doc)
        # Calculate c_td
        ctd: ndarray = X.data / (1 - self.b + self.b * rep / self.avgdl)
        data: ndarray = X.data * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X = X @ self._idf_diag
        return X


class BM25LCanonicalTransformer(BM25TransformerBase):
    """
    Canonical BM25L from Lv & Zhai (2011), as reviewed by Kamphuis et al.
    (ECIR 2020) and implemented by `bm25s`.

    Differs from :class:`BM25LTransformer` (rank_bm25-compatible) by dropping
    the extra ``tf_td`` multiplier. Like the original paper, the formula is
    applied for any (doc, term) pair: documents that do not contain a query
    term still contribute a constant baseline ``idf * (k1+1)*δ/(k1+δ)``. That
    baseline is sparse-incompatible, so :meth:`transform` only emits the
    ``tf > 0`` component; full retrieval scoring (including the absent-term
    baseline) is provided by ``BM25Vectorizer.score`` / ``rank``.
    """  # noqa

    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """Same IDF as BM25L: ln((N+1)/(n_t+0.5))."""
        idf: ndarray = np.log((n_samples + 1) / (df + 0.5))
        return np.maximum(idf, self.epsilon * float(np.mean(idf)))

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        Canonical BM25L score for term t with tf_td > 0 in document d:

                                       (k1 + 1) · (c_td + δ)
        BM25L(t, d) = idf(t) · ----------------------------------
                                          k1 + c_td + δ

        Documents with ``tf_td = 0`` would, under the canonical formula,
        receive ``idf · (k1+1)·δ / (k1+δ)``; that constant is omitted here
        for sparse compatibility. Use ``BM25Vectorizer.score`` to include it.
        """  # noqa
        if copy:
            X = X.copy()
        nnz_per_doc: ndarray = np.diff(X.indptr)
        dl: ndarray = X.sum(axis=1).A1
        rep: ndarray = np.repeat(dl, nnz_per_doc)
        ctd: ndarray = X.data / (1 - self.b + self.b * rep / self.avgdl)
        data: ndarray = (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X = X @ self._idf_diag
        return X


class BM25PlusTransformer(BM25TransformerBase):
    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """
        BM25+ inverse-document frequency (IDF):
                     N + 1
        idf(t) = ln(-------)
                      n_t
        where
          N   = total number of documents
          n_t = number of documents that contain term t
        """  # noqa
        idf: ndarray = np.log((n_samples + 1) / df)
        return np.maximum(idf, self.epsilon * float(np.mean(idf)))

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        BM25+ score for term t in document d:

                                        (k1 + 1) · tf_td
        BM25+(t, d) = idf(t) · ( -------------------------------------  + δ )
                                   k1 · (1 – b + b · |d|/avgdl) + tf_td

        where
          tf_td  = term-frequency of t in d
          |d|   = document length
          avgdl = average document length in the corpus
          k1, b, δ = BM25+ parameters

        addresses the penalization of long documents by adding a delta parameter after
        the term frequency normalization step, rather than inside it as BM25L does.
        """  # noqa
        if copy:
            X = X.copy()
        nnz_per_doc: ndarray = np.diff(X.indptr)
        dl: ndarray = X.sum(axis=1).A1
        rep: ndarray = np.repeat(dl, nnz_per_doc)
        data: ndarray = self.delta + (
            (X.data * (self.k1 + 1)) / (self.k1 * (1 - self.b + self.b * rep / self.avgdl) + X.data)
        )
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X = X @ self._idf_diag
        return X


class BM25AdptTransformer(BM25TransformerBase):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        use_idf: bool = True,
        max_iter: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(k1=k1, b=b, delta=delta, epsilon=epsilon, use_idf=use_idf, **kwargs)
        self.k1_terms: ndarray | None = None
        self.max_iter = max_iter
        self.info_gains: ndarray | None = None  # store G_1^q values

    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """
        BM25-adpt inverse-document frequency (IDF)

        BM25-adpt uses information gain (G_1^q) in place of IDF:
        G_1^q = log2((df_2 + 0.5)/(df_1 + 1)) - log2((df_1 + 0.5)/(N + 1))

        As a simplification for initial calculation, we use:
                         n_t + 0.5
        idf(t) = −log2(-------------)
                          N + 1

        where
          N   = total number of documents
          n_t = number of documents that contain term t

        information gain based IDF
        """  # noqa
        # This is a simplified version - the real G_1^q will be calculated in fit
        idf: ndarray = -np.log2((df + 0.5) / (n_samples + 1))
        return np.maximum(idf, self.epsilon * float(np.mean(idf)))

    def _compute_df_r(self, X: csr_matrix, r: int, df: ndarray) -> ndarray:
        """
        Compute df_r based on the definition in the paper:
        df_r = |D_t|_{c_td ≥ r-0.5}| for r > 1
        df_r = df_t for r = 1
        df_r = N for r = 0

        Where c_td is the normalized term frequency.
        """  # noqa
        if r == 0:
            return np.full(X.shape[1], X.shape[0])

        if r == 1:
            return df

        # For r > 1, calculate normalized term frequencies and count docs where c_td ≥ r-0.5
        dl = X.sum(axis=1).A1
        avgdl = np.mean(dl)

        # Initialize result array
        df_r = np.zeros(X.shape[1], dtype=np.int32)

        # Iterate through documents to calculate c_td for each term
        for i in range(X.shape[0]):
            doc_start, doc_end = X.indptr[i], X.indptr[i + 1]
            doc_length = dl[i]

            for j in range(doc_start, doc_end):
                term_idx = X.indices[j]
                tf = X.data[j]

                # Calculate c_td for this term occurrence
                c_td = tf / (1 - self.b + self.b * doc_length / avgdl)

                # Check if c_td meets the threshold
                if c_td >= r - 0.5:
                    df_r[term_idx] += 1

        return df_r

    def _compute_info_gain(
        self, X: csr_matrix, n_samples: int, df: ndarray, max_r: int = 5
    ) -> tuple[ndarray, list[ndarray]]:
        """
        Compute information gain values G_r^q for different r values.
        Return G_1^q and a list of all G_r^q/G_1^q values for optimization.
        """  # noqa
        dl = X.sum(axis=1).A1
        avgdl = np.mean(dl)
        nnz_per_doc = np.diff(X.indptr)
        rep_dl = np.repeat(dl, nnz_per_doc)
        ctd = X.data / (1 - self.b + self.b * rep_dl / avgdl)

        df_values = [np.full(X.shape[1], X.shape[0]), df]
        for r in range(2, max_r + 2):
            mask = ctd >= r - 0.5
            df_values.append(np.bincount(X.indices[mask], minlength=X.shape[1]))

        # Calculate G_r^q values
        g_values = []
        for r in range(max_r):
            # G_r^q = log2((df_{r+1} + 0.5)/(df_r + 1)) - log2((df_r + 0.5)/(N + 1))
            g_r = np.log2((df_values[r + 2] + 0.5) / (df_values[r + 1] + 1)) - np.log2(
                (df_values[r + 1] + 0.5) / (n_samples + 1)
            )
            g_values.append(g_r)

        # G_1^q will be used as IDF substitute
        G_1 = g_values[0]

        # Compute G_r^q/G_1^q for optimization
        g_normalized = [np.divide(g, G_1, out=np.zeros_like(g), where=G_1 != 0) for g in g_values]

        return G_1, g_normalized

    def _optimize_k1(self, g_normalized: list[ndarray], max_r: int = 5) -> ndarray:
        """
        Find term-specific k1 values that minimize the difference between
        normalized information gain and BM25's score function.

        k1' = arg min_{k1} sum_{r=1}^{max_r} (G_r^q/G_1^q - (k1+1)*r/(k1+r))^2
        """
        n_terms = g_normalized[0].shape[0]
        k1_terms = np.full(n_terms, self.k1)  # Initialize with default k1

        # Optimize for each term using a simple grid search for demonstration
        # In practice, Newton-Raphson or other optimization methods would be better
        for term_idx in range(n_terms):
            best_k1 = self.k1
            min_error = float("inf")

            # Try different k1 values and find the one that minimizes error
            for k1_candidate in np.linspace(0.1, 5.0, 50):  # Range of potential k1 values
                error = 0
                for r in range(1, max_r + 1):
                    # BM25 score component for comparison
                    bm25_score = (k1_candidate + 1) * r / (k1_candidate + r)

                    # Compare with normalized information gain
                    if r - 1 < len(g_normalized):
                        g_term = g_normalized[r - 1][term_idx]
                        error += (g_term - bm25_score) ** 2

                if error < min_error:
                    min_error = error
                    best_k1 = k1_candidate

            k1_terms[term_idx] = best_k1

        return k1_terms

    @override
    def fit(self, X: csr_matrix) -> Self:
        """
        Fit the transformer, calculating term-specific k1 values based on
        information gain as described in the paper.
        """  # noqa
        df: ndarray = np.bincount(X.indices, minlength=X.shape[1])
        n_samples: int = X.shape[0]
        self.avgdl = X.sum(axis=1).A1.mean()

        # Compute information gain values
        G_1, g_normalized = self._compute_info_gain(X, n_samples, df)

        # Store G_1^q to use as IDF
        self.info_gains = G_1
        self._idf_diag = sp.diags(G_1, offsets=0, format="csr")

        # Optimize for term-specific k1 values
        self.k1_terms = self._optimize_k1(g_normalized)
        return self

    @override
    def fit_from_stats(self, df: ndarray, n_samples: int, doc_len: ndarray) -> Self:
        raise NotImplementedError("BM25-adpt requires fitting on the full term-count matrix")

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        BM25-adpt score for term t in document d:

                                    (k1_t + 1) · tf_td
        BM25-adpt(t, d) = G_1^q · -------------------------------------
                                  k1_t · (1 – b + b · |d|/avgdl) + tf_td

        where
          k1_t  = term-specific TF-saturation parameter calculated from
                  information gain
          G_1^q = information gain at r=1, used in place of IDF
          tf_td = term frequency of t in d
          |d|   = document length
          avgdl = average document length in the corpus
          b     = length-normalisation parameter
        """  # noqa
        if self.k1_terms is None or self.info_gains is None:
            raise RuntimeError("fit() must be called before transform()")

        if copy:
            X = X.copy()

        # document lengths |d|
        nnz_per_doc = np.diff(X.indptr)  # shape = (n_docs,)
        dl = X.sum(axis=1).A1
        rep_dl = np.repeat(dl, nnz_per_doc)  # 1 per non-zero entry
        # term-specific k1 vector, 1 per non-zero entry
        k1_vec = self.k1_terms[X.indices]
        # BM25-adpt core
        denom = k1_vec * (1 - self.b + self.b * rep_dl / self.avgdl) + X.data
        data = X.data * (k1_vec + 1) / denom
        X_new = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X_new = X_new @ self._idf_diag
        return X_new


class BM25TTransformer(BM25TransformerBase):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        use_idf: bool = True,
        max_iter: int = 20,
        **kwargs,
    ) -> None:
        """
        :param max_iter: Max Newton-Raphson iterations
        """  # noqa
        super().__init__(k1=k1, b=b, delta=delta, epsilon=epsilon, use_idf=use_idf, **kwargs)
        self.k1_terms: Optional[ndarray] = None
        self.max_iter = max_iter

    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """
        BM25T inverse-document frequency (IDF):
                        N + 1
        idf(t) = ln(---------------)
                       n_t + 0.5
        where
          N   = total number of documents
          n_t = number of documents that contain term t
        """  # noqa
        idf: ndarray = np.log((n_samples + 1) / (df + 0.5))
        return np.maximum(idf, self.epsilon * float(np.mean(idf)))

    def _compute_g_k1(self, k1: float) -> float:
        """
        Compute g_{k1} as defined in the paper:
        g_{k1} = k1/(k1-1) * log(k1) if k1 != 1, else 1
        """  # noqa
        if abs(k1 - 1.0) < 1e-10:  # Close to 1, use the limit value
            return 1.0
        else:
            return (k1 / (k1 - 1.0)) * np.log(k1)

    def _compute_g_k1_derivative(self, k1: float) -> float:
        """
        Compute the derivative of g_{k1} for Newton-Raphson method
        """  # noqa
        if abs(k1 - 1.0) < 1e-10:  # Near k1=1, use numerical approximation
            return 0.5  # Approximation of the derivative at k1=1
        else:
            # Derivative of (k1/(k1-1))*log(k1)
            return np.log(k1) / (k1 - 1) - k1 * np.log(k1) / ((k1 - 1) ** 2) + 1 / (k1 - 1)

    def _compute_term_specific_k1(self, X: csr_matrix, df: ndarray) -> ndarray:
        """
        Compute term-specific k1 values using the log-logistic method described in the paper.
        Use Newton-Raphson method to solve for k1'.
        """  # noqa
        n_terms = X.shape[1]
        k1_terms = np.full(n_terms, self.k1)  # Initialize with default k1

        dl = X.sum(axis=1).A1
        norm = 1 - self.b + self.b * dl / self.avgdl
        X_csc = X.tocsc()

        # For each term, solve for term-specific k1
        for term_idx in range(n_terms):
            if df[term_idx] == 0:  # Skip terms that don't appear in corpus
                continue

            start, end = X_csc.indptr[term_idx], X_csc.indptr[term_idx + 1]
            if start == end:
                continue

            c_td = X_csc.data[start:end] / norm[X_csc.indices[start:end]]
            target = np.log(c_td + 1).sum() / df[term_idx]

            # Newton-Raphson method to solve for k1'
            k1_current = self.k1  # Start with default k1
            for _ in range(self.max_iter):
                g_k1 = self._compute_g_k1(k1_current)
                g_k1_derivative = self._compute_g_k1_derivative(k1_current)

                # Compute error and update
                error = g_k1 - target
                if abs(error) < 1e-6:  # Convergence check
                    break

                # Update k1 using Newton-Raphson step
                if abs(g_k1_derivative) > 1e-10:  # Avoid division by near-zero
                    k1_current = k1_current - error / g_k1_derivative
                    # Ensure k1 stays positive
                    k1_current = max(0.1, k1_current)
                else:
                    break  # Derivative too small, exit

            k1_terms[term_idx] = k1_current

        return k1_terms

    @override
    def fit(self, X: csr_matrix) -> Self:
        """
        Fit the transformer, calculating term-specific k1 values using
        the log-logistic method as described in the paper.
        """  # noqa
        df: ndarray = np.bincount(X.indices, minlength=X.shape[1])
        n_samples: int = X.shape[0]
        idf: ndarray = self._calc_idf(df, n_samples)
        self._idf_diag = sp.diags(idf, offsets=0, format="csr")
        self.avgdl = X.sum(axis=1).A1.mean()

        # Compute term-specific k1 values
        self.k1_terms = self._compute_term_specific_k1(X, df)
        return self

    @override
    def fit_from_stats(self, df: ndarray, n_samples: int, doc_len: ndarray) -> Self:
        raise NotImplementedError("BM25T requires fitting on the full term-count matrix")

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        BM25T score for term t in document d:

                                          tf_td · (k1_t + 1)
        BM25T(t, d) = idf(t) · ----------------------------------------
                                tf_td + k1_t · (1 – b + b · |d|/avgdl)

        where
          k1_t  = term-specific TF-saturation parameter calculated using
                  the log-logistic method
          tf_td = term frequency of t in d
          |d|   = document length
          avgdl = average document length in the corpus
          b     = length-normalisation parameter
        """  # noqa
        if copy:
            X = X.copy()
        # document lengths (|d|)
        dl: ndarray = X.sum(axis=1).A1
        nnz_per_doc: ndarray = np.diff(X.indptr)
        # length == nnz
        rep: ndarray = np.repeat(dl, nnz_per_doc)
        # Use term-specific k1 values
        k1_rep: ndarray = self.k1_terms[X.indices]
        # Apply BM25T formula with term-specific k1 values
        data: ndarray = X.data * (k1_rep + 1) / (X.data + k1_rep * (1 - self.b + self.b * rep / self.avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X = X @ self._idf_diag
        return X


class TFIDF1ApTransformer(BM25TransformerBase):
    @override
    def _calc_idf(self, df: ndarray, n_samples: int) -> ndarray:
        """
        TF1ap × IDF inverse-document frequency (IDF):
                     N + 1
        idf(t) = ln(-------)
                      n_t
        where
          N   = total number of documents
          n_t = number of documents that contain term t
        """  # noqa
        idf: ndarray = np.log((n_samples + 1) / df)
        return np.maximum(idf, self.epsilon * float(np.mean(idf)))

    @override
    def transform(self, X: csr_matrix, copy: bool = True) -> csr_matrix:
        """
        TF1ap × IDF score for term t in document d:
                                                          tf_td
        TF1ap(t, d) = idf(t) · ( 1 + ln(1 + ln( ------------------------ + δ )))
                                                 1 – b + b · |d| / avgdl
        where
          tf_td   = term frequency of t in d
          |d|    = document length
          avgdl  = average document length in the corpus
          b, δ   = TF1ap parameters
        """  # noqa
        if copy:
            X = X.copy()
        nnz_per_doc: ndarray = np.diff(X.indptr)
        dl: ndarray = X.sum(axis=1).A1
        rep: ndarray = np.repeat(dl, nnz_per_doc)
        data: ndarray = 1 + np.log(1 + np.log(X.data / (1 - self.b + self.b * rep / self.avgdl) + self.delta))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            X = X @ self._idf_diag
        return X


class SimilarityMixin:
    def similarity(
        self,
        item1: Any,
        item2: Any,
        metric: Literal["cosine", "dot", "jaccard"] = "cosine",
        transformer: str | None = None,
    ) -> float:
        """
        Compute similarity between *item1* and *item2*.

        Parameters
        ----------
        item1, item2 : str | Iterable[str] | sp.spmatrix
            Raw docs or pre-computed sparse row vectors.
        metric : {"cosine", "dot", "jaccard"}
            Similarity metric to use.
        transformer : str | None
            Weighting variant to apply ("bm25", "bm25l", …).  Defaults to the
            one this vectorizer was fitted with.

        Returns
        -------
        float
            Similarity in the range:
                – dot: [0, +∞)
                – cosine: [-1, 1] but scaled to [0, 1]  (0 if one vector is all zeros)
                – jaccard: [0, 1] (1 if both vectors are empty)
        """  # noqa
        check_is_fitted(self, "_tfidf")

        v1: sp.csr_matrix = self._vectorize(item1, transformer)
        v2: sp.csr_matrix = self._vectorize(item2, transformer)

        if metric == "dot":
            # (1 × n) · (n × 1)  →  (1 × 1) sparse matrix
            return float((v1 @ v2.T)[0, 0])

        if metric == "cosine":
            num = (v1 @ v2.T)[0, 0]
            den = sparse_norm(v1) * sparse_norm(v2)
            # If either vector is zero-norm, define cosine = 0 · (avoid /0)
            return 0.0 if den == 0 else float(num / den)

        if metric == "jaccard":
            idx1 = set(v1.indices)
            idx2 = set(v2.indices)
            if not idx1 and not idx2:  # both empty → define J = 1
                return 1.0
            inter = len(idx1 & idx2)
            union = len(idx1 | idx2)
            return inter / union

        raise ValueError(f"Unsupported metric '{metric}'. Choose from 'cosine', 'dot', or 'jaccard'.")


class BM25Vectorizer(TfidfVectorizer, SimilarityMixin):
    transformer_dispatch: dict[str, Type[BM25TransformerBase]] = {
        "bm25": BM25Transformer,
        "bm25l": BM25LTransformer,
        "bm25l_canonical": BM25LCanonicalTransformer,
        "bm25plus": BM25PlusTransformer,
        "bm25adpt": BM25AdptTransformer,
        "bm25t": BM25TTransformer,
        "tfidf1ap": TFIDF1ApTransformer,
    }

    def __init__(
        self,
        transformer: Literal[
            "bm25", "bm25l", "bm25l_canonical", "bm25plus", "bm25adpt", "bm25t", "tfidf1ap"
        ] = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        epsilon: float = 0.25,
        log1p_idf: bool = False,
        use_idf: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.transformer: str = transformer
        self.k1: float = k1
        self.b: float = b
        self.delta: float = delta
        self.epsilon: float = epsilon
        self.log1p_idf: bool = log1p_idf
        self.use_idf: bool = use_idf
        self._tfidf: BM25TransformerBase | None = None
        self._df: ndarray | None = None
        self._n_samples: int | None = None
        self._doc_len: ndarray | None = None
        self._count_matrix: csr_matrix | None = None

    def fit(self, raw_documents: list | np.ndarray, y: np.ndarray | None = None) -> Self:
        X: csr_matrix = CountVectorizer.fit_transform(self, raw_documents)
        self._fit_count_matrix(X)
        return self

    def _fit_count_matrix(self, X: csr_matrix) -> None:
        self._df = np.bincount(X.indices, minlength=X.shape[1])
        self._n_samples = X.shape[0]
        self._doc_len = X.sum(axis=1).A1
        self._count_matrix = X
        self._tfidf = self.transformer_dispatch[self.transformer](
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            epsilon=self.epsilon,
            log1p_idf=self.log1p_idf,
            use_idf=self.use_idf,
        ).fit(X)

    def transform(self, raw_documents: list | np.ndarray) -> csr_matrix:
        check_is_fitted(self, "_tfidf")
        counts = CountVectorizer.transform(self, raw_documents)
        return self._tfidf.transform(counts)

    def fit_transform(self, raw_documents: list | np.ndarray, y: np.ndarray | None = None) -> csr_matrix:
        X: csr_matrix = CountVectorizer.fit_transform(self, raw_documents)
        self._fit_count_matrix(X)
        return self._tfidf.transform(X)

    def _get_weighting_transformer(self, transformer: str | None = None) -> BM25TransformerBase:
        if transformer is None or transformer == self.transformer:
            return self._tfidf

        if self._df is None or self._n_samples is None or self._doc_len is None:
            raise RuntimeError("fit() must be called before using an alternative transformer")

        if transformer in {"bm25adpt", "bm25t"}:
            if self._count_matrix is None:
                raise RuntimeError(f"Alternative transformer '{transformer}' requires fitted corpus counts")
            return self.transformer_dispatch[transformer](
                k1=self.k1,
                b=self.b,
                delta=self.delta,
                epsilon=self.epsilon,
                log1p_idf=self.log1p_idf,
                use_idf=self.use_idf,
            ).fit(self._count_matrix)

        return self.transformer_dispatch[transformer](
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            epsilon=self.epsilon,
            log1p_idf=self.log1p_idf,
            use_idf=self.use_idf,
        ).fit_from_stats(self._df, self._n_samples, self._doc_len)

    def score(
        self,
        queries: str | Iterable[str] | sp.spmatrix,
        documents: Iterable[str] | sp.spmatrix | None = None,
        transformer: str | None = None,
    ) -> ndarray:
        """
        Compute direct query-document retrieval scores.

        Unlike cosine similarity over transformed sparse feature vectors, this
        sums each query term's contribution against each document. For BM25+,
        this includes the canonical delta contribution for query terms that are
        absent from a document.
        """
        check_is_fitted(self, "_tfidf")

        query_counts = self._count_documents(queries)
        document_counts = self._get_document_counts(documents)

        tf = self._get_weighting_transformer(transformer)
        if isinstance(tf, BM25PlusTransformer):
            return self._score_bm25plus(query_counts, document_counts, tf)
        if isinstance(tf, BM25LCanonicalTransformer):
            return self._score_bm25l_canonical(query_counts, document_counts, tf)

        document_weights = tf.transform(document_counts)
        return (query_counts @ document_weights.T).toarray()

    def _count_documents(self, documents: str | Iterable[str] | sp.spmatrix) -> csr_matrix:
        if sp.isspmatrix(documents):
            return documents.tocsr(copy=False)
        docs = [documents] if isinstance(documents, str) else list(documents)
        return CountVectorizer.transform(self, docs)

    def _get_document_counts(self, documents: Iterable[str] | sp.spmatrix | None = None) -> csr_matrix:
        if documents is None:
            if self._count_matrix is None:
                raise RuntimeError("fit() must be called before scoring fitted documents")
            return self._count_matrix
        return self._count_documents(documents)

    def _score_bm25plus(
        self,
        query_counts: csr_matrix,
        document_counts: csr_matrix,
        transformer: BM25PlusTransformer,
    ) -> ndarray:
        if query_counts.shape[1] != document_counts.shape[1]:
            raise ValueError("query and document matrices must have the same number of features")

        dl = document_counts.sum(axis=1).A1
        nnz_per_doc = np.diff(document_counts.indptr)
        rep = np.repeat(dl, nnz_per_doc)
        data = (document_counts.data * (transformer.k1 + 1)) / (
            transformer.k1 * (1 - transformer.b + transformer.b * rep / transformer.avgdl) + document_counts.data
        )
        component = sp.csr_matrix((data, document_counts.indices, document_counts.indptr), shape=document_counts.shape)

        if transformer.use_idf:
            idf = transformer._idf_diag.diagonal()
            component = component @ transformer._idf_diag
            baseline = query_counts @ (idf * transformer.delta)
        else:
            baseline = query_counts.sum(axis=1).A1 * transformer.delta

        scores = (query_counts @ component.T).toarray()
        return scores + np.asarray(baseline).reshape(-1, 1)

    def _score_bm25l_canonical(
        self,
        query_counts: csr_matrix,
        document_counts: csr_matrix,
        transformer: "BM25LCanonicalTransformer",
    ) -> ndarray:
        if query_counts.shape[1] != document_counts.shape[1]:
            raise ValueError("query and document matrices must have the same number of features")

        document_weights = transformer.transform(document_counts)
        scores = (query_counts @ document_weights.T).toarray()

        tfc_absent = (transformer.k1 + 1) * transformer.delta / (transformer.k1 + transformer.delta)
        if transformer.use_idf:
            idf = transformer._idf_diag.diagonal()
            weight = idf * tfc_absent
        else:
            weight = np.full(document_counts.shape[1], tfc_absent)

        baseline = np.asarray(query_counts @ weight).reshape(-1, 1)

        has_term = (document_counts > 0).astype(np.float64)
        weighted_query = query_counts.multiply(weight)
        correction = (weighted_query @ has_term.T).toarray()

        return scores + baseline - correction

    def rank(
        self,
        queries: str | Iterable[str] | sp.spmatrix,
        documents: Iterable[str] | sp.spmatrix | None = None,
        transformer: str | None = None,
        top_n: int | None = None,
        return_scores: bool = False,
        batch_size: int = 4096,
    ) -> ndarray | tuple[ndarray, ndarray]:
        """
        Rank documents for each query using direct retrieval scores.

        Returns document indices sorted by descending score. If `return_scores`
        is True, also returns the sorted scores with the same shape.
        """
        check_is_fitted(self, "_tfidf")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        query_counts = self._count_documents(queries)
        document_counts = self._get_document_counts(documents)
        n_queries = query_counts.shape[0]
        n_documents = document_counts.shape[0]
        if top_n is None:
            top_n = n_documents
        if not 0 < top_n <= n_documents:
            raise ValueError("top_n must be between 1 and the number of documents")

        tf = self._get_weighting_transformer(transformer)
        needs_special_score = isinstance(tf, (BM25PlusTransformer, BM25LCanonicalTransformer))
        document_weights = None if needs_special_score else tf.transform(document_counts)
        ranked_batches = []
        score_batches = []

        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            query_batch = query_counts[start:end]
            if isinstance(tf, BM25PlusTransformer):
                score_batch = self._score_bm25plus(query_batch, document_counts, tf)
            elif isinstance(tf, BM25LCanonicalTransformer):
                score_batch = self._score_bm25l_canonical(query_batch, document_counts, tf)
            else:
                score_batch = (query_batch @ document_weights.T).toarray()

            if top_n == n_documents:
                ranked_batch = np.argsort(score_batch, axis=1)[:, ::-1]
            else:
                candidate_idx = np.argpartition(score_batch, -top_n, axis=1)[:, -top_n:]
                candidate_scores = np.take_along_axis(score_batch, candidate_idx, axis=1)
                order = np.argsort(candidate_scores, axis=1)[:, ::-1]
                ranked_batch = np.take_along_axis(candidate_idx, order, axis=1)

            ranked_batches.append(ranked_batch)
            if return_scores:
                score_batches.append(np.take_along_axis(score_batch, ranked_batch, axis=1))

        ranked = np.vstack(ranked_batches)
        if not return_scores:
            return ranked
        return ranked, np.vstack(score_batches)

    def _vectorize(
        self,
        item: Any,
        transformer: str | None = None,
    ) -> sp.csr_matrix:
        """
        Convert *item* to a single-row sparse weighted vector.

        Parameters
        ----------
        item : str | Iterable[str] | sp.spmatrix
            – raw document (str),
            – iterable of raw documents (first element is taken), or
            – pre-computed sparse row vector.
        transformer : str | None
            If given, use this weighting variant instead of the one fitted on
            `self.transformer`.  The new transformer is created lazily and
            cached per call; use when you need, e.g., BM25+ on top of a BM25
            model without refitting the main vectorizer.
        """  # noqa
        # Already a sparse vector → just ensure shape is (1, n_features)
        if sp.isspmatrix(item):
            if item.shape[0] != 1:
                raise ValueError("Sparse item must be a single-row vector")
            return item.tocsr(copy=False)

        # Normalise to *one* raw document string
        if isinstance(item, str):
            docs: list[str] = [item]
        elif isinstance(item, Iterable):
            docs_list = list(item)
            if not docs_list or not isinstance(docs_list[0], str):
                raise TypeError("Iterable *item* must contain at least one raw-string document")
            docs = [docs_list[0]]
        else:
            raise TypeError("item must be a raw document (str), an iterable[str], or a scipy sparse row-vector")

        counts = CountVectorizer.transform(self, docs)

        tf = self._get_weighting_transformer(transformer)
        return tf.transform(counts)
