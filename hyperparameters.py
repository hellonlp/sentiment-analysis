# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:23:12 2018

@author: cm
"""

import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)

class Hyperparamters:
    # Train parameters   
    print_step = 100
    batch_size = 16       
    
    # Optimization parameters
    num_train_epochs = 20
    warmup_proportion = 0.1
    use_tpu = None
    do_lower_case = True    
    learning_rate = 5e-5 

    # TextCNN parameters
    num_filters = 128 
    filter_sizes = [2,3,4,5,6,7]
    embedding_size = 384
    keep_prob = 0.5
    
    # Sequence and Label
    sequence_length = 60 
    num_labels = 3
        
    # BERT parameters
    name = 'albert_small_zh_google'
    bert_path = os.path.join(pwd,name)
    data_dir = os.path.join(pwd,'data')
    vocab_file = os.path.join(pwd,name,'vocab_chinese.txt')   
    init_checkpoint = os.path.join(pwd,name,'albert_model.ckpt')
    saved_model_path = os.path.join(pwd,'model')    
    
    
    
    
    
    
    
    


    
    
