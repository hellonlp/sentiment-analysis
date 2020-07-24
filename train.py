# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:42:07 2019

@author: cm
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import tensorflow as tf
from sentiment_analysis_albert.classifier_utils import get_features,get_features_test
from sentiment_analysis_albert.networks import NetworkAlbert
from sentiment_analysis_albert.hyperparameters import Hyperparamters as hp
from sentiment_analysis_albert.utils import select,shuffle_one,time_now_string


pwd = os.path.dirname(os.path.abspath(__file__))
MODEL = NetworkAlbert(is_training=True )



# 训练集
input_ids,input_masks,segment_ids,label_ids = get_features()
input_ids_test,input_masks_test,segment_ids_test,label_ids_test = get_features_test()
print('1'*10)
# 训练数据和测试数据
N = len(input_ids)
N_train = N
ids_train = np.arange(N_train)               
num_batches = int((N_train - 1) /hp.batch_size) + 1
print('number of batch:',num_batches)
# 测试集
ids_test = np.arange(len(input_ids_test)) 
# 启动图和训练
saver = tf.train.Saver(max_to_keep=100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 恢复模型参数
MODEL_SAVE_PATH = os.path.join(pwd,'model')
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
     print('Restored model!')


with sess.as_default():
    for i in range(hp.num_train_epochs):
        indexs = shuffle_one(ids_train)       
        for batch_num in range(num_batches-1):
            i1 = indexs[batch_num * hp.batch_size:min((batch_num + 1) * hp.batch_size, N_train)]
            # Get features
            input_id_ = select(input_ids,i1)
            input_mask_ = select(input_masks,i1)
            segment_id_ = select(segment_ids,i1)
            label_id_ = select(label_ids,i1)
            # Feed dict
            fd = {MODEL.input_ids: input_id_,
                  MODEL.input_masks: input_mask_,
                  MODEL.segment_ids:segment_id_,
                  MODEL.label_ids:label_id_}
            # Optimizer            
            sess.run(MODEL.optimizer, feed_dict = fd)       
            # Save Model
            if batch_num%600==0:
                print ('epoch:',i,'batch_num:',batch_num)
                saver.save(sess, os.path.join(pwd, 'model', 'model_%s_%s.ckpt'%(str(i),str(batch_num))))
            # Print
            if batch_num % hp.print_step == 0:
                # Loss of Train data 
                fd = {MODEL.input_ids: input_id_,
                      MODEL.input_masks: input_mask_ ,
                      MODEL.segment_ids:segment_id_,
                      MODEL.label_ids:label_id_}
                loss = sess.run(MODEL.loss, feed_dict = fd)
                print('Time:',time_now_string(),'Epoch:',i,'Batch number:',batch_num,'Loss(train):',loss)
                # Loss of Test data
                indexs_test = shuffle_one(ids_test)[:128]
                input_id_test = select(input_ids_test,indexs_test)
                input_mask_test = select(input_masks_test,indexs_test)
                segment_id_test = select(segment_ids_test,indexs_test)
                label_id_test = select(label_ids_test,indexs_test)
                fd_test = {MODEL.input_ids:input_id_test,
                           MODEL.input_masks:input_mask_test ,
                           MODEL.segment_ids:segment_id_test,
                           MODEL.label_ids:label_id_test}
                loss = sess.run(MODEL.loss, feed_dict = fd_test)
                print('Time:',time_now_string(),'Epoch:',i,'Batch number:',batch_num,'Loss(test):',loss)                                          
    print('Optimization finished')




