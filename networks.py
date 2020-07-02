# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:44:42 2019

@author: cm
"""

import os
import tensorflow as tf
from sentiment_analysis_albert import modeling
from sentiment_analysis_albert.modules import cell_textcnn
from sentiment_analysis_albert.hyperparameters import Hyperparamters as hp
from sentiment_analysis_albert import optimization
from sentiment_analysis_albert.classifier_utils import ClassifyProcessor



num_labels = hp.num_labels
processor = ClassifyProcessor() 
bert_config_file = os.path.join(hp.bert_path,'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)




class NetworkAlbert(object):
    def __init__(self,is_training):
        self.is_training = is_training
        self.input_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_ids')
        self.input_masks = tf.placeholder(tf.int32, shape=[None,  hp.sequence_length], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None,  hp.sequence_length], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, shape=[None], name='label_ids')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        
        
        #加载BERT词向量
        self.model = modeling.AlbertModel(
                                    config=bert_config,
                                    is_training=self.is_training,
                                    input_ids=self.input_ids,
                                    input_mask=self.input_masks,
                                    token_type_ids=self.segment_ids,
                                    use_one_hot_embeddings=False)

        ### 获取3D向量：（batch_size,sequence_length,hidden_size）
        output_layer_init = self.model.get_sequence_output()#(4,30,768)
     
        # hidden_size
        hidden_size = output_layer_init.shape[-1].value
          
        # cell cnn
        output_layer = cell_textcnn(output_layer_init,self.is_training)
        # hidden_size new
        hidden_size = output_layer.shape[-1].value         
        #                           
        output_weights = tf.get_variable(
              "output_weights", [num_labels, hidden_size],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
        
        output_bias = tf.get_variable(
              "output_bias", [num_labels], initializer=tf.zeros_initializer())
        
        with tf.variable_scope("loss"):
            if self.is_training:
              # I.e., 0.1 dropout
              output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)#(4,768)
        
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)#(4,2) 
            self.probabilities = tf.nn.softmax(logits, axis=-1)#(4,1)
            
            # Prediction
            self.preds = tf.argmax(logits, axis=-1, output_type=tf.int32)     
                       
            #
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = \
                     modeling.get_assignment_map_from_checkpoint(tvars,
                                                                 hp.init_checkpoint)
            tf.train.init_from_checkpoint(hp.init_checkpoint, assignment_map)

            if self.is_training:
                log_probs = tf.nn.log_softmax(logits, axis=-1)
            
                one_hot_labels = tf.one_hot(self.label_ids, depth=num_labels, dtype=tf.float32)
            
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)

                # Optimizer BERT
                train_examples = processor.get_train_examples(hp.data_dir)
                num_train_steps = int(
                    len(train_examples) / hp.batch_size * hp.num_train_epochs)
                num_warmup_steps = int(num_train_steps * hp.warmup_proportion)
                print('num_train_steps',num_train_steps)
                self.optimizer = optimization.create_optimizer(self.loss,
                                                                hp.learning_rate, 
                                                                num_train_steps, 
                                                                num_warmup_steps,
                                                                hp.use_tpu,
                                                                )            
  


if __name__ == '__main__':
    #模型加载
    albert = NetworkAlbert(is_training=True)



