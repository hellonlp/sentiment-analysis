# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:44:08 2020

@author: cm
"""

import os
from sentiment_analysis_dict.utils import ToolGeneral


pwd = os.path.dirname(os.path.abspath(__file__))
tool = ToolGeneral()    


class Hyperparams:
    '''Hyper parameters'''
    # Load sentiment dictionary
    deny_word = tool.load_dict(os.path.join(pwd,'dict','not.txt'))
    posdict = tool.load_dict(os.path.join(pwd,'dict','positive.txt'))
    negdict = tool.load_dict(os.path.join(pwd,'dict', 'negative.txt'))
    pos_neg_dict = posdict|negdict
    # Load adverb dictionary
    mostdict = tool.load_dict(os.path.join(pwd,'dict','most.txt'))
    verydict = tool.load_dict(os.path.join(pwd,'dict','very.txt'))
    moredict = tool.load_dict(os.path.join(pwd,'dict','more.txt'))
    ishdict = tool.load_dict(os.path.join(pwd,'dict','ish.txt'))
    insufficientlydict = tool.load_dict(os.path.join(pwd,'dict','insufficiently.txt'))
    overdict = tool.load_dict(os.path.join(pwd,'dict','over.txt'))
    inversedict = tool.load_dict(os.path.join(pwd,'dict','inverse.txt'))




