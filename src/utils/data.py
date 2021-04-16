from wget import download
import os
import tarfile
import re
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary, MmCorpus

nltk.download('wordnet')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ROOT = os.path.expanduser("~/dpfa-imdb")
DATA_DIR = os.path.join(ROOT, "datasets")
IMDB_SOURCE = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
NEWSGROUP_SOURCE = "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz"
PATTERN = re.compile(r"\d+_([0-9]|10)\.txt")
LEMMATIZER = WordNetLemmatizer()
IMDB_DECOMPRESS_FOLDER = "aclImdb"
NEWSGROUP_DECOMPRESS_FOLDER = "20news-18828"


def pull_from_url(path, dataset="imdb"):
    if dataset == "imdb":
        url = IMDB_SOURCE
        decompress_folder = IMDB_DECOMPRESS_FOLDER
    if dataset == "20newsgroup":
        url = NEWSGROUP_SOURCE
        decompress_folder = NEWSGROUP_DECOMPRESS_FOLDER
    filename = url.split("/")[-1]
    if not os.path.exists(os.path.join(path, filename)):
        logging.info("Pulling {} dataset from URL: {}".format(dataset.upper(), url))
        download(url, out=path)
    else:
        logging.info("File aleady exists, skip download!")
    if not os.path.exists(os.path.join(path, decompress_folder)):
        with tarfile.open(os.path.join(path, filename), "r:gz") as gz:
            logging.info("Extracting data from compressed file...")
            gz.extractall(path=path)
    else:
        logging.info("Compressed file was already extracted in current folder!")


def load_imdb(
        root=DATA_DIR,
        train=True,
        download=False
):
    data_folder = os.path.join(root, "datasets/imdb")
    processed_folder = os.path.join(data_folder, "processed")
    subfolder = "train" if train else "test"
    if download:
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        pull_from_url(data_folder)
        docs = []
        ratings = []
        labels = []
        for cls in ["pos", "neg"]:
            work_dir = os.path.join(data_folder, IMDB_DECOMPRESS_FOLDER, subfolder, cls)
            filenames = os.listdir(work_dir)
            for file in filenames:
                with open(os.path.join(work_dir, file), "r", encoding="utf8") as f:
                    docs.append(word_tokenize(f.read().strip()))
                    ratings.append(PATTERN.search(file).group(1))
                    labels.append(1 if cls == "pos" else 0)
        docs = [[LEMMATIZER.lemmatize(token) for token in doc] for doc in docs]
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
        if train:
            logging.info("Building dictionary and BOW corpus for training data...")
            corpus, dictionary = build_corpus(docs)
            MmCorpus.serialize(os.path.join(processed_folder, "train_corpus.mm"), corpus)
            dictionary.save(os.path.join(processed_folder, "dictionary"))
        else:
            logging.info("Building BOW corpus for testing data...")
            dictionary = Dictionary.load(os.path.join(processed_folder, "dictionary"))
            corpus, _ = build_corpus(docs, dictionary)
            MmCorpus.serialize(os.path.join(processed_folder, "test_corpus.mm"), corpus)
        return corpus, dictionary
    else:
        try:
            if train:
                return MmCorpus(os.path.join(processed_folder, "train_corpus.mm")), \
                       Dictionary.load(os.path.join(processed_folder, "dictionary")),
            else:
                return MmCorpus(os.path.join(processed_folder, "test_corpus.mm")), \
                       Dictionary.load(os.path.join(processed_folder, "dictionary"))
        except FileNotFoundError:
            logging.warning("The dataset does not exist, please set download to True!")
            return None, None


def load_20newsgroup(
        root=DATA_DIR,
        download=False
):
    data_folder = os.path.join(root, "datasets/20newsgroup")
    processed_folder = os.path.join(data_folder, "processed")
    if download:
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        pull_from_url(data_folder, dataset="20newsgroup")
        docs = []
        groups = []
        subfolders = os.listdir(os.path.join(data_folder, NEWSGROUP_DECOMPRESS_FOLDER))
        for subf in subfolders:
            work_dir = os.path.join(data_folder, NEWSGROUP_DECOMPRESS_FOLDER, subf)
            filenames = os.listdir(work_dir)
            for file in filenames:
                with open(os.path.join(work_dir, file), "r") as f:
                    docs.append(word_tokenize(f.read().strip()))
                    groups.append(subf)
        docs = [[LEMMATIZER.lemmatize(token) for token in doc] for doc in docs]
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
        logging.info("Building dictionary and BOW corpus for the dataset...")
        corpus, dictionary = build_corpus(docs)
        MmCorpus.serialize(os.path.join(processed_folder, "corpus.mm"), corpus)
        dictionary.save(os.path.join(processed_folder, "dictionary"))
        return corpus, dictionary
    else:
        try:
            return MmCorpus(os.path.join(processed_folder, "corpus.mm")), \
                   Dictionary.load(os.path.join(processed_folder, "dictionary"))
        except FileNotFoundError:
            logging.warning("The dataset does not exist, please set download to True!")
            return None, None


def build_corpus(docs, dictionary=None):
    if dictionary is None:
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=50, no_above=0.3)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return corpus, dictionary


if __name__ == "__main__":
    load_imdb(root="./", download=True)
    load_20newsgroup(root="./", download=True)
