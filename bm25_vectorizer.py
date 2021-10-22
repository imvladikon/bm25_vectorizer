#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    _document_frequency,
)
from sklearn.preprocessing import normalize
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


class BM25Transformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        k1=2.0,
        b=0.75,
        delta=2.0
    ):

        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.k1 = k1
        self.b = b
        self.delta = delta

    def fit(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, **_astype_copy_false(df))

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = self.get_idf(document_frequency=df, n_samples=n_samples)
            # TODO: collect words with negative idf to set them a special epsilon value.
            # idf can be negative if word is contained in more than half of documents
            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def get_idf(self, document_frequency, n_samples):
        return np.log(n_samples - document_frequency + 0.5) - np.log(
            document_frequency + 0.5
        )

    def transform(self, X, copy=True):
        X = self._validate_data(
            X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

            # Document length (number of terms) in each row
            # Shape is (n_samples, 1)
            dl = X.sum(axis=1)
            # Number of non-zero elements in each row
            # Shape is (n_samples, )
            sz = X.indptr[1:] - X.indptr[0:-1]
            # In each row, repeat `dl` for `sz` times
            # Shape is (sum(sz), )
            # Example
            # -------
            # dl = [4, 5, 6]
            # sz = [1, 2, 3]
            # rep = [4, 5, 5, 6, 6, 6]
            rep = np.repeat(np.asarray(dl), sz)
            # Average document length
            # Scalar value
            avgdl = np.average(dl)
            # Compute BM25 score only for non-zero elements
            data = (
                X.data
                * (self.k1 + 1)
                / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
            )
            X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"], msg="idf vector is not fitted")

            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        """Inverse document frequency vector, only defined if `use_idf=True`.

        Returns
        -------
        ndarray of shape (n_features,)
        """
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(
            value, diags=0, m=n_features, n=n_features, format="csr"
        )

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse"]}


class BM25LTransformer(BM25Transformer):
    def __init__(
        self,
        *,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        k1=2.0,
        b=0.75,
        delta=2.0
    ):

        super().__init__(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            k1=k1,
            b=b,
            delta=delta,
        )

    def fit(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, **_astype_copy_false(df))

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = self.get_idf(document_frequency=df, n_samples=n_samples)
            # TODO: collect words with negative idf to set them a special epsilon value.
            # idf can be negative if word is contained in more than half of documents
            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def get_idf(self, document_frequency, n_samples):

        return np.log(n_samples + 1) - np.log(document_frequency + 0.5)

    def transform(self, X, copy=True):
        X = self._validate_data(
            X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

            # Document length (number of terms) in each row
            # Shape is (n_samples, 1)
            dl = X.sum(axis=1)
            # Number of non-zero elements in each row
            # Shape is (n_samples, )
            sz = X.indptr[1:] - X.indptr[0:-1]
            # In each row, repeat `dl` for `sz` times
            # Shape is (sum(sz), )
            # Example
            # -------
            # dl = [4, 5, 6]
            # sz = [1, 2, 3]
            # rep = [4, 5, 5, 6, 6, 6]
            rep = np.repeat(np.asarray(dl), sz)
            # Average document length
            # Scalar value
            avgdl = np.average(dl)
            # Compute BM25 score only for non-zero elements
            ctd = X.data / (1 - self.b + self.b * rep / avgdl)
            data = (
                X.data * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
            )
            X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"], msg="idf vector is not fitted")

            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X


class BM25PlusTransformer(BM25Transformer):
    def __init__(
        self,
        *,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        k1=2.0,
        b=0.75,
        delta=2.0
    ):

        super().__init__(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            k1=k1,
            b=b,
            delta=delta,
        )

    def fit(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, **_astype_copy_false(df))

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = self.get_idf(document_frequency=df, n_samples=n_samples)
            # TODO: collect words with negative idf to set them a special epsilon value.
            # idf can be negative if word is contained in more than half of documents
            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def get_idf(self, document_frequency, n_samples):

        return np.log(n_samples + 1) - np.log(document_frequency)

    def transform(self, X, copy=True):
        X = self._validate_data(
            X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

            # Document length (number of terms) in each row
            # Shape is (n_samples, 1)
            dl = X.sum(axis=1)
            # Number of non-zero elements in each row
            # Shape is (n_samples, )
            sz = X.indptr[1:] - X.indptr[0:-1]
            # In each row, repeat `dl` for `sz` times
            # Shape is (sum(sz), )
            # Example
            # -------
            # dl = [4, 5, 6]
            # sz = [1, 2, 3]
            # rep = [4, 5, 5, 6, 6, 6]
            rep = np.repeat(np.asarray(dl), sz)
            # Average document length
            # Scalar value
            avgdl = np.average(dl)
            # Compute BM25 score only for non-zero elements
            ctd = X.data / (1 - self.b + self.b * rep / avgdl)
            data = (
                X.data
                * (self.k1 + 1)
                * (self.delta + (X.data * (self.k1 + 1)))
                / (self.k1 * (1 - self.b + self.b * rep / avgdl) + X.data)
            )

            X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"], msg="idf vector is not fitted")

            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X


class BM25Vectorizer(TfidfVectorizer):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        transformer="bm25",
        k1=2.0,
        b=0.75,
        delta=0.5
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        transformer_dispatch = {
            "bm25": BM25Transformer,
            "bm25l": BM25LTransformer,
            "bm25plus": BM25PlusTransformer,
        }

        self._tfidf = transformer_dispatch.get(transformer, BM25Transformer)(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            k1=k1,
            b=b,
            delta=delta,
        )


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = BM25Vectorizer()
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names_out())
    print(X.data)

    vectorizer = BM25Vectorizer(transformer="bm25l")
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names_out())
    print(X.data)

    vectorizer = BM25Vectorizer(transformer="bm25plus")
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names_out())
    print(X.data)
