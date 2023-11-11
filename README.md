## BM25 vectorizer

sklearn compatible bm25 vectorizers

```bash
pip install git+https://github.com/imvladikon/bm25_vectorizer -q
```

```python
from bm25_vectorizer import BM25Vectorizer

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
```

## References:

Based on (with some modifications):
[rank_bm25](https://github.com/dorianbrown/rank_bm25)
