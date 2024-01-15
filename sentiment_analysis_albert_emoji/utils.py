# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:40:40 2020

@author: cm
"""



import time
import emoji
import pandas as pd
import numpy as np


  
def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime( time.time() )) 


def get_emoji(sentence):
    emoji_list = emoji.emoji_lis(sentence)
    return ''.join([l['emoji'] for l in emoji_list])


def load_csv(file,header=0,encoding="utf-8"):
    return pd.read_csv(file,
                       encoding=encoding,
                       header=header,
                       error_bad_lines=False)


def save_csv(dataframe,file,header=True,index=None,encoding="utf-8"):
    return dataframe.to_csv(file,
                            mode='w+',
                            header=header,
                            index=index,
                            encoding=encoding)


def save_excel(dataframe,file,header=True,sheetname='Sheet1'):
    return dataframe.to_excel(file,
                         header=header,
                         sheet_name=sheetname) 
    
    

def load_excel(file,header=0,sheetname=None):
	dfs = pd.read_excel(file,
                     header=header,
                     sheet_name=sheetname)
	sheet_names = list(dfs.keys())
	print('Name of first sheet:',sheet_names[0])
	df = dfs[sheet_names[0]]
	print('Load excel data finished!')
	return df.fillna("")  
    

def load_txt(file):
    with  open(file,encoding='utf-8',errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
    return lines


def save_txt(file,lines):
    lines = [l+'\n' for l in lines]
    with  open(file,'w+',encoding='utf-8') as fp:#a+添加
        fp.writelines(lines)
    

def shuffle_two(a1,a2):
    """
    随机打乱a1和a2两个
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    a2_ = [a2[l] for l in ran]
    return a1_, a2_


def load_vocabulary(file_vocabulary_label):
    """
    Load vocabulary to dict
    """
    vocabulary = load_txt(file_vocabulary_label)
    dict_id2char,dict_char2id = {},{}
    for i,l in enumerate(vocabulary):
        dict_id2char[i] = str(l)
        dict_char2id[str(l)] = i
    return dict_id2char,dict_char2id


def get_word_sequence(words,vocabulary,Reverse=True,k=1000):
    """
    words: a list of word or string
    """
    words = [l.lower() for l in words]    
    dic = {}
    for word in words:
        if word not in dic:
            dic[word] = 1
        else:
            dic[word] = dic[word] + 1
    return sorted(dic.items(),key = lambda x:x[0],reverse = Reverse)[:k]     


def select(data,ids):
    return [data[i] for i in ids]


def shuffle_one(a1):
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    return np.array(a1)[ran].tolist()


def cut_list(data,size):
    """
    data: a list
    size: the size of cut
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data)-1)//size + 1)]


def cut_list_by_size(data,lengths):
    """
    data: a list
    lengths: the different sizes of cut
    """
    list_block = []
    for l in lengths:
        list_block.append(data[:l])
        data = data[l:]
    return list_block



if __name__ == "__main__": 
    print(time_now_string())    
    
    #
    string = '☕️🥂'
    print(get_emoji(string))
    #
    file_vocab_emoji ='albert_base_zh/vocab_emoji.txt'
    vocab = load_txt(file_vocab_emoji)
    print(vocab[-3:])
    
    
    
    
    
    
    
    
    
    
    