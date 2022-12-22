# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:40:40 2019

@author: cm
"""

import time
import pandas as pd
import numpy as np



def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime( time.time() )) 


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
    

def select(data,ids):
    return [data[i] for i in ids]

def shuffle_one(a1):
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    return a1_


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
    dic = {}
    for i,l in enumerate(vocabulary):
        dic[str(i)] = str(l)
    return dic


if __name__ == "__main__": 
    print(time_now_string())    

    
    
    
    
    
    
    
    
    
    
    