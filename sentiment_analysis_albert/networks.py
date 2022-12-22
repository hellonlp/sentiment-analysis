# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:44:42 2019

@author: cm
"""

import os
import tensorflow as tf
from sentiment_analysis_albert import modeling,optimization
from sentiment_analysis_albert.classifier_utils import ClassifyProcessor
from sentiment_analysis_albert.modules import cell_textcnn
from sentiment_analysis_albert.hyperparameters import Hyperparamters as hp
from sentiment_analysis_albert.utils import time_now_string


num_labels = hp.num_labels
processor = ClassifyProcessor() 
bert_config_file = os.path.join(hp.bert_path,'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)


def count_model_params():
    """
    Compte the parameters
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('  + Number of params: %.2fM' % (total_parameters / 1e6))
        
    
class NetworkAlbert(object):
    def __init__(self,is_training):
        self.is_training = is_training
        self.input_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_ids')
        self.input_masks = tf.placeholder(tf.int32, shape=[None,  hp.sequence_length], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None,  hp.sequence_length], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, shape=[None], name='label_ids')        
        # Load BERT Pre-training LM
        self.model = modeling.AlbertModel(
                                    config=bert_config,
                                    is_training=self.is_training,
                                    input_ids=self.input_ids,
                                    input_mask=self.input_masks,
                                    token_type_ids=self.segment_ids,
                                    use_one_hot_embeddings=False)
        
        # Get the feature vector with size 3D：（batch_size,sequence_length,hidden_size）
        output_layer_init = self.model.get_sequence_output()                    
        # Cell textcnn
        output_layer = cell_textcnn(output_layer_init,self.is_training)        
        # Hidden size
        hidden_size = output_layer.shape[-1].value         
        # Dense
        with tf.name_scope("Full-connection"):                       
            output_weights = tf.get_variable(
                  "output_weights", [num_labels, hidden_size],
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
            
            output_bias = tf.get_variable(
                  "output_bias", [num_labels], initializer=tf.zeros_initializer())
            # Logit           
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            self.logits = tf.nn.bias_add(logits, output_bias)
            self.probabilities = tf.nn.softmax(self.logits, axis=-1)
        # Prediction
        with tf.variable_scope("Prediction"):                        
            self.preds = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
        # Summary for tensorboard        
        with tf.variable_scope("Loss"):                        
            if self.is_training:
	            self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.label_ids)))
	            tf.summary.scalar('Accuracy', self.accuracy) 
                
            # Check whether has loaded model
            ckpt = tf.train.get_checkpoint_state(hp.saved_model_path)
            checkpoint_suffix = ".index"        
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
                print('='*10,'Restoring model from checkpoint!','='*10)
                print("%s - Restoring model from checkpoint ~%s" % (time_now_string(),
                                                                    ckpt.model_checkpoint_path))
            else:                   
                # Load BERT Pre-training LM
                print('='*10,'First time load BERT model!','='*10)
                tvars = tf.trainable_variables()
                if hp.init_checkpoint:
                   (assignment_map, initialized_variable_names) = \
                     modeling.get_assignment_map_from_checkpoint(tvars,
                                                                 hp.init_checkpoint)
                   tf.train.init_from_checkpoint(hp.init_checkpoint, assignment_map)
            
            # Optimization  
            if self.is_training:                
                # Global_step
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # Loss                
                log_probs = tf.nn.log_softmax(self.logits, axis=-1)            
                one_hot_labels = tf.one_hot(self.label_ids, depth=num_labels, dtype=tf.float32)                
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)                
                # Optimizer
                train_examples = processor.get_train_examples(hp.data_dir)
                num_train_steps = int(
                    len(train_examples) / hp.batch_size * hp.num_train_epochs)
                num_warmup_steps = int(num_train_steps * hp.warmup_proportion)
                self.optimizer = optimization.create_optimizer(self.loss,
                                                                hp.learning_rate, 
                                                                num_train_steps, 
                                                                num_warmup_steps,
                                                                hp.use_tpu,
                                                                Global_step=self.global_step,
                                                                )  
                # Summary for tensorboard                 
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
                
        # Compte the parameters
        count_model_params()
        vs = tf.trainable_variables()
        for l in vs:
            print(l) 




if __name__ == '__main__':
    # Load model
    albert = NetworkAlbert(is_training=True)


