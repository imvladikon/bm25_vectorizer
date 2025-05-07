## BM25 vectorizer

BM25 Transformers and Vectorizer
This Python package provides implementations of BM25, BM25L, BM25+, BM25-adpt, BM25T, and TF₁ₐₚ × IDF ranking functions
as scikit-learn compatible transformers and a vectorizer. These are used for information retrieval and text processing,
extending the traditional TF-IDF approach with document length normalization and term frequency saturation.

### Installation

```bash
pip install git+https://github.com/imvladikon/bm25_vectorizer -q
```

### Usage

Similar to tf-idf from sklearn,

```pycon
BM25Vectorizer(transformer="bm25plus").fit(corpus)
```

where `transformer` can be one of the following: `bm25l`, `bm25plus`, `bm25adpt`, `bm25t`, `tfidf1ap`

#### Feature Extraction

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

#### Similarity Calculation

```python
from bm25_vectorizer import BM25Vectorizer

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over the lazy dog quickly"
]

vec = BM25Vectorizer(transformer="bm25plus").fit(corpus)

print(vec.similarity("quick brown fox", "lazy dog", metric="cosine"))
print(vec.similarity("fox lazy", "lazy fox", metric="jaccard"))
```

#### Ranking

```python
from bm25_vectorizer import BM25Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = BM25Vectorizer()
X = vectorizer.fit_transform(corpus)
query = 'first document'
query_vector = vectorizer.transform([query])
similarity = cosine_similarity(X, query_vector)
ranked_indices = np.argsort(similarity.flatten())[::-1]
print("Ranked documents:", ranked_indices)
```

#### Classes

* BM25TransformerBase: Abstract base class for BM25 transformers.
* BM25Transformer: Implements the standard BM25 scoring function.
* BM25LTransformer: Implements BM25L, which adjusts for document length more aggressively.
* BM25PlusTransformer: Implements BM25+, which adds a constant boost to scores.
* BM25AdptTransformer: Implements BM25-adpt, using term-specific $k_1^t$ via information gain.
* BM25TTransformer: Implements BM25T, using term-specific $k_1^t$ via log-logistic estimation.
* TFIDFTransformer: Implements TF₁ₐₚ × IDF, using logarithmic term frequency transformation.
* BM25Vectorizer: Combines CountVectorizer with a BM25 transformer for end-to-end text processing.

#### Parameters

* k1: Controls term frequency saturation (float, default: 1.5).
* b: Controls document length normalization (float, default: 0.75).
* delta: Additional parameter for BM25L, BM25+, and TF₁ₐₚ × IDF (float, default: 1.0).
* epsilon: Minimum IDF value to prevent negative IDFs (float, default: 0.25).
* use_idf: Whether to apply IDF weighting (bool, default: True).

### BM25 Formulas

Below are the formulas for the BM25 variants implemented in this package, provided for validation:

* ATIRE BM25 IDF: $\text{idf}(t) = \log\left(\frac{N}{n(t)}\right)$

* ATIRE BM25
  Score: $\text{BM25}(t,d) = \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$
* Standard BM25 IDF: $\text{idf}(t) = \log\left(\frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$
* Standard BM25
  Score: $\text{BM25}(t,d) = \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$
* BM25L IDF: $\text{idf}(t) = \log\left(\frac{N + 1}{n(t) + 0.5}\right)$
* BM25L
  Score: $\text{BM25L}(t,d) = \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1) \cdot (c(t,d) + \delta)}{k_1 + c(t,d) + \delta}$
  where $c(t,d) = \frac{f(t,d)}{1 - b + b \cdot \frac{|d|}{\text{avgdl}}}$

* BM25+ IDF: $\text{idf}(t) = \log\left(\frac{N + 1}{n(t)}\right)$

* BM25+
  Score: $\text{BM25+}(t,d) = \text{IDF}(t) \cdot \left( \delta + \frac{f(t,d) \cdot (k_1 + 1)}{k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}}) + f(t,d)} \right)$

* BM25-adpt IDF: $\text{idf}(t) = -\log_2\left(\frac{n(t) + 0.5}{N + 1}\right)$

* BM25-adpt
  Score: $\text{BM25-adpt}(t,d) = \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1^t + 1)}{k_1^t \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}}) + f(t,d)}$
  where $k_1^t$ is a term-specific parameter computed via information gain.

* BM25T IDF: $\text{idf}(t) = \log\left(\frac{N + 1}{n(t) + 0.5}\right)$

* BM25T
  Score: $\text{BM25T}(t,d) = \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1^t + 1)}{f(t,d) + k_1^t \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$
  where $k_1^t$ is a term-specific parameter computed via log-logistic estimation.

* TF1ap × IDF IDF: $\text{idf}(t) = \ln\left(\frac{N + 1}{n(t)}\right)$

* TF1ap × IDF
  Score: $\text{TF1ap}(t,d) = \text{IDF}(t) \cdot \left(1 + \ln\left(1 + \ln\left(\frac{f(t,d)}{1 - b + b \cdot \frac{|d|}{\text{avgdl}}} + \delta\right)\right)\right)$

##### Notation

$N$: Total number of documents.    
$n(t)$: Number of documents containing term $t$.    
$f(t,d)$: Frequency of term $t$ in document $d$.    
$|d|$: Length of document $d$.    
$\text{avgdl}$: Average document length across the collection.   
$k_1$: Term frequency saturation parameter (default: 1.5).   
$k_1^t$: Term-specific saturation parameter for BM25-adpt and BM25T.    
$b$: Document length normalization parameter (default: 0.75).     
$\delta$: Additional parameter for BM25L, BM25+, and TF₁ₐₚ × IDF (default: 1.0).     
$\epsilon$: IDF smoothing parameter (default: 0.25).

## References

- https://github.com/dorianbrown/rank_bm25

- http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
  "Improvements to BM25 and Language Models Examined", Trotman et al.
- https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html