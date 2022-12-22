# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:12:10 2020

@author: cm
"""

import numpy as np
from sentiment_analysis_bayes.utils import load_txt, drop_stopwords
from sentiment_analysis_bayes.hyperparameters import Hyperparamters as hp

# Load data
vocabulary = [str(w.replace('\n', '')) for w in load_txt(hp.file_vocabulary)][:hp.feature_size]
stopwords = set(load_txt(hp.file_stopwords))


def get_sentence_feature(sentence):
    """
    Transform a sentence to one-hot vector.
    """
    words = drop_stopwords(sentence, stopwords)
    return [int(words.count(w)) for w in vocabulary]


def load_label(file_train_label):
    """
    Load data label.
    """
    return np.array([int(line) for line in load_txt(file_train_label)])


def load_feature(file_train_feature):
    """
    Load data one-hot feature.
    """
    return np.array([eval(line) for line in load_txt(file_train_feature)])


if __name__ == '__main__':
    # 
    train_label = load_label(hp.file_train_label)
    print(train_label[:5])
    # 
    train_feature = load_feature(hp.file_test_feature)
    print(train_feature[0][:20])
