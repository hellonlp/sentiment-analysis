# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:42:41 2020

@author: cm
"""

import os

pwd = os.path.dirname(os.path.abspath(__file__))


class Hyperparamters:
    # Parameters
    feature_size = 2000

    # Stopwords
    file_stopwords = os.path.join(pwd, 'dict/stopwords.txt')
    file_vocabulary = os.path.join(pwd, 'dict/vocabulary_pearson_40000.txt')

    # Train data
    file_train_data = os.path.join(pwd, 'data/train.csv')
    file_test_data = os.path.join(pwd, 'data/test.csv')
    #
    file_train_feature = os.path.join(pwd, 'data/train_feature.txt')
    file_train_label = os.path.join(pwd, 'data/train_label.txt')
    #
    file_test_feature = os.path.join(pwd, 'data/test_feature.txt')
    file_test_label = os.path.join(pwd, 'data/test_label.txt')

    # model file
    file_p0 = os.path.join(pwd, 'model/p0.txt')
    file_p1 = os.path.join(pwd, 'model/p1.txt')
    file_class = os.path.join(pwd, 'model/class.txt')
