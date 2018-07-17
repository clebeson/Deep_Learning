from abc import ABCMeta, abstractmethod 
import pickle
import numpy as np
import os
import sys
import tensorflow as tf
import numpy as np
import os.path
from base.basemodel import BaseModel
from  base.hyperparameters import Hyperparameters
from itertools import izip as zip
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base.functions as bf
import layers


class Empty(BaseModel):
    def __init__(self, dataset, name = "empty"):

        BaseModel.__init__(self,
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "",
                                    "file_name" : name,
                                    "url" : "",
                                    "model_name" : name  
                                    },
                       dataset = dataset, cut_layer = None
                      )
        tf.reset_default_graph()
       
        h, w = self.hparams.height, self.hparams.width
        if self.hparams.data_augmentation and not self.hparams.auto_crop == (0,0):
            h = self.hparams.height 
            w = self.hparams.width 
        input_data = tf.placeholder(tf.float32, shape=[None,h, w, self.hparams.channels], name='images')
        labels = tf.placeholder(tf.float32, shape=[None, self.hparams.num_classes], name='labels')
        self.dict_model["images"] = input_data 
        self.dict_model["labels"] = labels
        self.dict_model["keep"] = tf.placeholder(tf.float32, name='dropout_keep')
        self.dict_model["istraining"] = tf.placeholder(tf.float32, name='istraining')
        
    
    def _create_cnn(self, inputs, batch_norm = None, pretrained_weights = None): pass
       
    
    def _create_fully_connected(self, batch_norm = None): pass
    
    def build(self, is_tf = True, new_fc = True, istraining = False):
            
#             self.close()
#             tf.reset_default_graph()

            self._istraining = istraining
#             if not istraining:
#                 self.dataset.set_tfrecord(False)
   
#             if  not new_fc:
#                 self.hparams.cut_layer = self.info["last_layer"]
#                 self.hparams.fine_tunning = True
                
#             with tf.device("cpu:0"):
#                 if self.dataset.istfrecord():
#                     input_data, labels = self.dataset.get_tfrecord_data()
#                     self.hparams.bottleneck = False
    #                     input_images =  self.input_data
                
#             print("***********************************") 
#             print(self.dict_model)
#             g = tf.get_default_graph()
#             for i in g.get_operations():
#                 print(i.values())

            
            if istraining:
                self.generate_optimization_parameters()
                
        
            if self.sess is None:
                self.sess, self._saver = self.create_monitored_session(self.dict_model)
       
            for l in self._layers:
                l._model_input_data_placeholder = self.dict_model["images"]       
                l._labels_placeholder = self.dict_model["labels"]
                l._keep_placeholder= self.dict_model["keep"]
                l._istraining_placeholder = self.dict_model["istraining"]
                l._sess = self.sess
                l._logits = self._layers[-1]
            
            
           
            self.built = True
        
        
        
        
