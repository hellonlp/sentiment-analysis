# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:42:07 2020

@author: cm
"""



import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import tensorflow as tf

from sentiment_analysis_albert_emoji.classifier_utils import get_features
from sentiment_analysis_albert_emoji.classifier_utils import get_features_test
from sentiment_analysis_albert_emoji.networks import NetworkAlbert
from sentiment_analysis_albert_emoji.hyperparameters import Hyperparamters as hp
from sentiment_analysis_albert_emoji.utils import shuffle_one,select
from sentiment_analysis_albert_emoji.utils import time_now_string
from sentiment_analysis_albert_emoji.classifier_utils import get_features_emoji
from sentiment_analysis_albert_emoji.classifier_utils import get_features_emoji_test



# Load Model
pwd = os.path.dirname(os.path.abspath(__file__))
MODEL = NetworkAlbert(is_training=True)

# Get data features
input_ids,input_masks,segment_ids,label_ids = get_features()
input_ids_test,input_masks_test,segment_ids_test,label_ids_test = get_features_test()
num_train_samples = len(input_ids)
arr = np.arange(num_train_samples)               
num_batchs = int((num_train_samples - 1)/hp.batch_size) + 1
print('number of batch:',num_batchs)
ids_test = np.arange(len(input_ids_test)) 


# Get emoji features
input_emojis = get_features_emoji()
input_emojis_test = get_features_emoji_test()


# Set up the graph
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load model saved before
MODEL_SAVE_PATH = os.path.join(pwd, hp.file_save_model)
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
     print('Restored model!')


with sess.as_default():
    # Tensorboard writer
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    for i in range(hp.num_train_epochs):
        indexs = shuffle_one(arr)       
        for j in range(num_batchs-1):
            i1 = indexs[j * hp.batch_size:min((j + 1) * hp.batch_size, num_train_samples)]
            
            # Get features
            input_id_ = select(input_ids,i1)
            input_mask_ = select(input_masks,i1)
            segment_id_ = select(segment_ids,i1)
            label_id_ = select(label_ids,i1)
                   
            # Get features emoji
            input_emoji_ = select(input_emojis,i1)
                                      
            # Feed dict
            fd = {MODEL.input_ids: input_id_,
                  MODEL.input_masks: input_mask_,
                  MODEL.segment_ids:segment_id_,
                  MODEL.label_ids:label_id_,
                  MODEL.input_emojis:input_emoji_}
            
            # Optimizer            
            sess.run(MODEL.optimizer, feed_dict = fd)
            
            # Tensorboard
            if j%hp.summary_step==0:
                summary,glolal_step = sess.run([MODEL.merged,MODEL.global_step], feed_dict = fd)
                writer.add_summary(summary, glolal_step)
          
            # Save Model
            if j%(num_batchs//hp.num_saved_per_epoch)==0:
                if not os.path.exists(os.path.join(pwd, hp.file_save_model)):
                    os.makedirs(os.path.join(pwd, hp.file_save_model)) 
                saver.save(sess, os.path.join(pwd, hp.file_save_model, 'model_%s_%s.ckpt'%(str(i),str(j))))
            
            # Log
            if j % hp.print_step == 0:
                loss = sess.run(MODEL.loss, feed_dict = fd)      
                print('Time:%s, Epoch:%s, Batch number:%s/%s, Loss:%s'%(time_now_string(),str(i),str(j),str(num_batchs),str(loss)))           
                #  Loss of Test data
                indexs_test = shuffle_one(ids_test)[:hp.batch_size_eval]
                input_id_test = select(input_ids_test,indexs_test)
                input_mask_test = select(input_masks_test,indexs_test)
                segment_id_test = select(segment_ids_test,indexs_test)
                label_id_test = select(label_ids_test,indexs_test)
                
                # Get features emoji
                input_emoji_test = select(input_emojis_test,indexs_test)
                
                fd_test = {MODEL.input_ids:input_id_test,
                           MODEL.input_masks:input_mask_test ,
                           MODEL.segment_ids:segment_id_test,
                           MODEL.label_ids:label_id_test,
                           MODEL.input_emojis:input_emoji_test}
                loss = sess.run(MODEL.loss, feed_dict = fd_test)           
                print('Time:%s, Epoch:%s, Batch number:%s/%s, Loss(test):%s'%(time_now_string(),str(i),str(j),str(num_batchs),str(loss)))           
    print('Optimization finished')






