# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:23:12 2018

@author: cm
"""



import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
print('pwd:',pwd)


class Hyperparamters:
    # Train parameters
    num_train_epochs = 5
    print_step = 10 
    batch_size = 64     
    batch_size_eval = 128 
    summary_step = 10
    num_saved_per_epoch = 3
    max_to_keep = 100

    # Model paths
    logdir = 'logdir/model_01'
    file_save_model = 'model/model_01'
    file_load_model = 'model/model_01'

    # Train data and test data
    train_data = "sa_train.csv"
    test_data = "sa_test.csv"
    
    # Optimization parameters
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
    dict_label = {
    '0': '-1',
    '1': '0',
    '2': '1'}

    
    # ALBERT parameters
    name = 'albert_small_zh_google'
    bert_path = os.path.join(pwd,name)
    data_dir = os.path.join(pwd,'data')
    vocab_file = os.path.join(pwd,name,'vocab_chinese.txt')  
    init_checkpoint = os.path.join(pwd,name,'albert_model.ckpt')
    saved_model_path = os.path.join(pwd,'model')    
    
    
    
    
    
    
    
    


    
    
