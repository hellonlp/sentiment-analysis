# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:50:55 2020

@author: cm
"""

import numpy as np
from sentiment_analysis_bayes.bayes import save_model
from sentiment_analysis_bayes.load import load_feature, load_label
from sentiment_analysis_bayes.hyperparameters import Hyperparamters as hp


def main(x, y):
    """
    Training 
    """
    numTrainDocs = len(x)
    numWords = len(x[0])
    pAbusive = sum(y) / float(numTrainDocs)
    p0Num, p1Num = np.ones(numWords), np.ones(numWords)
    p0Deom, p1Deom = 2, 2
    for i in range(numTrainDocs):
        if y[i] == 1:
            p1Num = p1Num + x[i]
            p1Deom = p1Deom + sum(x[i])
        else:
            p0Num = p0Num + x[i]
            p0Deom = p0Deom + sum(x[i])
        if i % 100 == 0:
            print(i)
    p1Vect = p1Num / p1Deom
    p0Vect = p0Num / p0Deom
    p1VectLog = np.zeros(len(p1Vect))
    for i in range(len(p1Vect)):
        p1VectLog[i] = np.log(p1Vect[i])
    p0VectLog = np.zeros(len(p0Vect))
    for i in range(len(p0Vect)):
        p0VectLog[i] = np.log(p0Vect[i])
    return p0VectLog, p1VectLog, pAbusive


if __name__ == '__main__':
    # Train
    train_data, train_label = load_feature(hp.file_train_feature), load_label(hp.file_train_label)
    p0, p1, class_ = main(train_data, train_label)
    # Save model
    f1 = 'model/p0.txt'
    f2 = 'model/p1.txt'
    f3 = 'model/class.txt'
    save_model(p0, p1, class_, f1, f2, f3)
