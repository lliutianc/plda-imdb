import re, pickle
import tarfile
from smart_open import open
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from scipy.sparse import save_npz

import nltk
nltk.download('punkt')

IMDB_SOURCE = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


def generate_docs_from_url(url=IMDB_SOURCE):
    with open(url, 'rb') as infile:
        with tarfile.open(fileobj=infile) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.match(r'aclImdb/(train|test)/(pos|neg)/\d+_([0-9]|10)\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf8', errors='replace')


vectorizer = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words='english',
    max_df=0.3,
    min_df=5
)

bow = vectorizer.fit_transform(list(generate_docs_from_url())) # (50000, 27918)

save_npz('./imdb_bow.npz', bow)
with open('./vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
