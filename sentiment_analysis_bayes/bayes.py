# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:40:26 2018

@author: cm
"""

import os
import numpy as np
from sentiment_analysis_bayes.utils import load_txt, save_txt
from sentiment_analysis_bayes.hyperparameters import Hyperparamters as hp

pwd = os.path.dirname(os.path.abspath(__file__))


def classify(vec2classify, p0, p1, class_):
    """
    Classifier function of Bayes.
    """
    p1 = sum(vec2classify * p1) + np.log(class_)
    p0 = sum(vec2classify * p0) + np.log(1 - class_)
    if p1 > p0:
        return 1
    else:
        return 0


def load_model():
    """
    Load bayes parameters
    """
    p0 = np.array([float(l) for l in load_txt(hp.file_p0)])
    p1 = np.array([float(l) for l in load_txt(hp.file_p1)])
    class_ = float(load_txt(hp.file_class)[0])
    return p0, p1, class_


def save_model(p0, p1, class_, file_p0, file_p1, file_class):
    """
    Save bayes parameters
    """
    save_txt(file_p0, [str(l) for l in p0])
    save_txt(file_p1, [str(l) for l in p1])
    save_txt(file_class, [str(class_)])
    print('Save model finished!')


if __name__ == '__main__':
    # Predict
    print('我爱武汉')
