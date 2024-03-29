# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:01:45 2020

@author: cm
"""


import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper

from sentiment_analysis_albert_emoji.hyperparameters import Hyperparamters as hp



def cell_embedding(inputs):
    with tf.device('/cpu:0'):
        W = tf.Variable(tf.random_uniform([hp.vocab_size_emoji, hp.embedding_size], -1.0, 1.0), name="W_embedding")
        embedded_input = tf.nn.embedding_lookup(W,inputs)   
    return embedded_input


def cell_textcnn_two(inputs,is_training):
    # Add a dimension in the end：-1
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
                                        ksize=[1, hp.sequence_length + hp.sequence_length_emoji - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        with tf.name_scope("Concat"):
            num_filters_total = hp.num_filters * len(hp.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # Dropout
    h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, keep_prob=hp.keep_prob if is_training else 1)
    return h_pool_flat_dropout

