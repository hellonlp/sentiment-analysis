# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:01:45 2019

@author: cm
"""


import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from sentiment_analysis_albert.hyperparameters import Hyperparamters as hp



def cell_lstm(inputs,hidden_size,is_training):
    """
    inputs shape: (batch_size,sequence_length,embedding_size)
    hidden_size: rnn hidden size
    """
    with tf.variable_scope('cell_lstm'):
  
        cell_forward = tf.contrib.rnn.BasicLSTMCell(hidden_size/2)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(hidden_size/2)
        cell_forward = DropoutWrapper(cell_forward, 
                                      input_keep_prob=1.0, 
                                      output_keep_prob=0.5 if is_training else 1)
        cell_backward = DropoutWrapper(cell_backward, 
                                       input_keep_prob=1.0, 
                                       output_keep_prob=0.5 if is_training else 1)                
        
        print('cell_forward: ',cell_forward )
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_forward,
                                                          cell_backward,
                                                          inputs,
                                                          dtype=tf.float32)
        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)
        # 激活函数
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)            
        value = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0] - 1))            
        return last#(?,768)



def cell_textcnn(inputs,is_training):
    # 最后一个维度增加：-1
    inputs_expand = tf.expand_dims(inputs, -1)
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    with tf.name_scope("TextCNN"):
        for i, filter_size in enumerate(hp.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, hp.embedding_size, 1, hp.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                b = tf.Variable(tf.constant(0.1, shape=[hp.num_filters]),dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                                    inputs_expand,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                                        h,
                                        ksize=[1, hp.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = hp.num_filters * len(hp.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # Dropout
    h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, keep_prob=hp.keep_prob if is_training else 1)
    return h_pool_flat_dropout
            




