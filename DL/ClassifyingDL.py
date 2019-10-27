# Author: Mahboobeh Ghalehnovi and Arash Rahnama
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import io
import os
import scipy
import matplotlib
import sklearn
import tensorflow as tf
FLAGS = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import numpy as np
from sklearn import decomposition
from datetime import datetime
#FLAGS = None
import operator
import pandas as pd
import time
import timeit
from sklearn.metrics import accuracy_score
import csv
import sys
import gc


n_hidden_1 = 1000 # 1st layer number of neurons
n_hidden_2 = 600 # 2nd layer number of neurons
n_hidden_3 = 320 # 3rd layer number of neurons
n_hidden_4 = 170 # 4th layer number of neurons
n_hidden_5 = 85 # 5th layer number of neurons
n_hidden_6 = 40 # 5th layer number of neurons
n_hidden_7 = 12 # 6th layer number of neurons

def DLTF(dat_train,labs_train,dat_test,labs_test,batch_size,num_classes,Num_Epochs,LR):
    ROOT_PATH = os.getcwd()
    
    def neural_net(x_dict):
      with tf.name_scope("data"):
        x = x_dict['data']
        dat_flat = tf.contrib.layers.flatten(x)
      layer_1 = tf.layers.dense(dat_flat, n_hidden_1, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                               bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='input-layer')
      layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='hidden-layer1')
      layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='hidden-layer2')
      layer_4 = tf.layers.dense(layer_3, n_hidden_4, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='hidden-layer3')
      layer_5 = tf.layers.dense(layer_4, n_hidden_5, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='hidden-layer4')
      layer_6 = tf.layers.dense(layer_5, n_hidden_6, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='hidden-layer5')
      layer_7 = tf.layers.dense(layer_6, n_hidden_7, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='hidden-layer6')
      out_layer = tf.layers.dense(layer_7, num_classes, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                                bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None), bias_regularizer=tf.contrib.layers.l2_regularizer(scale = 0.01,scope=None),name='output-layer')
      tf.summary.histogram('hist_outputlayer',out_layer)
      
      return out_layer

    def model_fn(features, labels, mode):
    
        #with tf.device('/device:GPU:3'):
           # print(3)
        logits = neural_net(features) 
        with tf.name_scope("predictions"):
            pred_classes = tf.argmax(logits, axis=1)
            pred_probas = tf.nn.softmax(logits)
            
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
        
        with tf.name_scope("loss"):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)), name = "loss")
            tf.summary.scalar('train_loss', loss_op)
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(LR)
            train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
        
        with tf.name_scope("accuracy"):
            acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes, name = "accuracy")
            tf.summary.scalar('train_accuracy', tf.reduce_mean(acc_op))
        
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'predictions': pred_classes},
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'test_accuracy': acc_op})
        
        return estim_specs

    
    model = tf.estimator.Estimator(model_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(x={'data': dat_train}, y= labs_train, batch_size=batch_size,num_epochs=Num_Epochs, shuffle=False)
    model.train(input_fn)
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'data': dat_test}, y=labs_test,batch_size=batch_size,num_epochs=1,shuffle=False) 
    e = model.evaluate(input_fn)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={'data': dat_test}, y=labs_test, num_epochs=1, shuffle=False)
    predictions =list(model.predict(input_fn=predict_input_fn))
    acc_each_fold=accuracy_score(labs_test, predictions)
    del model
    gc.collect()
    return acc_each_fold
