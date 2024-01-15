# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2020

@author: cm
"""


import os
import sys
import tensorflow as tf
pwd = os.path.dirname(os.path.abspath(__file__))
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from sentiment_analysis_albert_emoji.networks import NetworkAlbert
from sentiment_analysis_albert_emoji.hyperparameters import Hyperparamters as hp
from sentiment_analysis_albert_emoji.classifier_utils import get_feature_test
from sentiment_analysis_albert_emoji.classifier_utils import get_feature_emoji_test



class ModelAlbertTextCNN(object):
    """
    Load NetworkAlbert TextCNN model
    """
    def __init__(self,):
        self.albert, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert =  NetworkAlbert(is_training=False)
                saver = tf.train.Saver()  
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd,hp.file_load_model))#small-google-gelu
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert,sess

MODEL = ModelAlbertTextCNN()
print('Load model finished!')



def get_sa(sentence):
    """
    Prediction of the sentence's sentiment.
    """
    feature = get_feature_test(sentence)
    feature_emoji = get_feature_emoji_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids:[feature[2]],
          MODEL.albert.input_emojis:[feature_emoji]
          }
    output = MODEL.sess.run(MODEL.albert.preds, feed_dict=fd)                               
    return output[0]-1
 


if __name__ == '__main__':
    ##
    sentence ='拍照效果：👍'
    print ("情感分析结果：",get_sa(sentence))


 