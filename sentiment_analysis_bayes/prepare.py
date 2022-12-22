# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:38:31 2020

@author: cm
"""

from sentiment_analysis_bayes.utils import drop_stopwords
from sentiment_analysis_bayes.utils import load_txt, load_csv, save_txt
from sentiment_analysis_bayes.hyperparameters import Hyperparamters as hp

# Load data
vocabulary = [str(w.replace('\n', '')) for w in load_txt(hp.file_vocabulary)][:hp.feature_size]
stopwords = set(load_txt(hp.file_stopwords))


def sentence2feature(sentence):
    """
    Transform a sentence to a One-hot vector.
    """
    words = drop_stopwords(sentence, stopwords)
    return [words.count(w) for w in vocabulary]


if __name__ == '__main__':
    # Train data
    df = load_csv(hp.file_train_data)
    contents = df['content'].tolist()
    labels = df['label'].tolist()
    train_features = [str(sentence2feature(l)) for l in contents]
    save_txt('data/train_feature.txt', train_features)
    train_labels = [str(l) for l in labels]
    save_txt('data/train_label.txt', train_labels)

    # Test data
    df = load_csv(hp.file_test_data)
    contents = df['content'].tolist()
    labels = df['label'].tolist()
    test_features = [str(sentence2feature(l)) for l in contents]
    save_txt('data/test_feature.txt', test_features)
    test_labels = [str(l) for l in labels]
    save_txt('data/test_label.txt', test_labels)
