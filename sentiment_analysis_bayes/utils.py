# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:45:17 2020

@author: cm
"""

import time
import jieba
import numpy as np
import pandas as pd


def time_now_string():
    """
    Time now.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def cut_list(data, size):
    """
    data: a list
    size: the size of cut
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data) - 1) // size + 1)]


def load_txt(file):
    """
    load a txt.
    """
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
    return lines


def save_txt(file, lines):
    """
    Save a txt.
    """
    lines = [l + '\n' for l in lines]
    with  open(file, 'w+', encoding='utf-8') as fp:
        fp.writelines(lines)


def drop_stopwords(sentence, stopwords):
    """
    Delete stopwords that we don't need.
    """
    return [l for l in jieba.lcut(str(sentence)) if l not in stopwords]


def load_csv(file, header=0, encoding="utf-8-sig"):
    """
    Load a Data-frame from a csv.
    """
    return pd.read_csv(file,
                       encoding=encoding,
                       header=header,
                       error_bad_lines=False)


def save_csv(dataframe, file, header=True, index=None, encoding="utf-8-sig"):
    """
    Save a Data-frame by a csv.
    """
    return dataframe.to_csv(file,
                            mode='w+',
                            header=header,
                            index=index,
                            encoding=encoding)


def shuffle_two(a1, a2):
    """
    Shuffle two lists by the same index.
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    return [a1[l] for l in ran], [a2[l] for l in ran]
