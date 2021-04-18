import re, pickle
import tarfile
from smart_open import open
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from scipy.sparse import save_npz

import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

IMDB_SOURCE = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
PATTERN = re.compile(r'aclImdb/(train|test)/(pos|neg)/\d+_([0-9]|10)\.txt')

labels = []


def generate_docs_from_url(url=IMDB_SOURCE, pat=PATTERN):
    with open(url, 'rb') as infile:
        with tarfile.open(fileobj=infile) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.match(r'aclImdb/(train|test)/(pos|neg)/\d+_([0-9]|10)\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    labels.append(pat.search(member.name).group(2))
                    yield member_bytes.decode('utf8', errors='replace')


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

    docs = list(generate_docs_from_url())[:(n_train + n_test)]
    docs_tr = tf_vectorizer.fit_transform(docs[:n_train])
    docs_te = tf_vectorizer.transform(docs[n_train:])
    return tf_vectorizer, docs_tr, docs_te


if __name__ == '__main__':
    vectorizer = CountVectorizer(
        tokenizer=word_tokenize,
        stop_words='english',
        max_df=0.3,
        min_df=5
    )

    bow = vectorizer.fit_transform(list(generate_docs_from_url())) # (50000, 41545)
    labels = list(map(lambda x: '1\n' if x == 'pos' else '0\n', labels))
    with open("./labels.txt", "w") as f:
        f.writelines(labels)
    save_npz('./imdb_bow.npz', bow)
    with open('./vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)