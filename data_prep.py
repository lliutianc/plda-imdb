import re, pickle
import tarfile
from smart_open import open
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from scipy.sparse import save_npz
import numpy as np

import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

IMDB_SOURCE = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
PATTERN = re.compile(r'aclImdb/(train|test)/(pos|neg)/\d+_([0-9]|10)\.txt')


def generate_docs_from_url(url=IMDB_SOURCE, pat=PATTERN):
    with open(url, 'rb') as infile:
        with tarfile.open(fileobj=infile) as tar:
            for member in tar.getmembers():
                if member.isfile() and pat.match(member.name):
                    member_bytes = tar.extractfile(member).read()
                    label = int(pat.search(member.name).group(2) == 'pos')
                    yield member_bytes.decode('utf8', errors='replace'), label


STOPS = set(stopwords.words('english'))


def is_stopword(word):
    VALID = re.compile('^[a-zA-Z]{2,}$')
    return word in STOPS or len(word) <= 2 or not bool(VALID.match(word))


def analyzer(tokenizer):
    def analyze(doc):
        doc_list = tokenizer(doc.lower().replace('\\', ''))
        return [word for word in doc_list if not is_stopword(word)]

    return analyze


def prepare_sparse_matrix(n_train, n_test, max_vocab_size, max_df=.5, min_df=10, tokenize=None):
    tokenize = tokenize or RegexpTokenizer('\w+|\$[\d\.]+|\S+').tokenize

    analyze = analyzer(tokenize)

    tf_vectorizer = CountVectorizer(
        analyzer=analyze,
        max_df=max_df,
        min_df=min_df,
        max_features=max_vocab_size,
    )
    docs = []
    labels = []
    for doc, lab in generate_docs_from_url():
        docs.append(doc)
        labels.append(lab)
    np.random.seed(1234)
    indices = np.random.choice(len(docs), size=n_train + n_test, replace=False)
    docs_tr = [docs[idx] for idx in indices[:n_train]]
    docs_te = [docs[idx] for idx in indices[n_train:]]
    labels_tr = np.array([labels[idx] for idx in indices[:n_train]])
    labels_te = np.array([labels[idx] for idx in indices[n_train:]])
    docs_tr = tf_vectorizer.fit_transform(docs_tr)
    docs_te = tf_vectorizer.transform(docs_te)
    return tf_vectorizer, docs_tr, docs_te, labels_tr, labels_te


if __name__ == '__main__':
    prepare_sparse_matrix(800, 200, 3000)
