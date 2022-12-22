# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:46:08 2019

@author: cm
"""

import os
import sys

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
from sentiment_analysis_bayes.bayes import classify
from sentiment_analysis_bayes.bayes import load_model
from sentiment_analysis_bayes.load import get_sentence_feature

p0Vec, p1Vec, pClass1 = load_model()


def sa(sentence):
    """
    Predict a sentence's sentiment.
    """
    vector = get_sentence_feature(sentence)
    point = classify(vector, p0Vec, p1Vec, pClass1)
    if point == 1:
        return 'Positif'
    elif point == 0:
        return 'Negitif'


if __name__ == '__main__':
    # Test
    content = '我喜欢武汉'
    content = '我讨厌武汉'
    print(sa(content))
