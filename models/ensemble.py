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
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from enum import Enum
import layers

class EnsembleType(Enum):
    DECAF = 1
    SUM = 2
    AVG = 3
    CONV3D = 4

class Ensemble(BaseModel):
    def __init__(self, dataset,  type, multiple_inputs = False):
        dataset.hparams.cut_layer = ""
        BaseModel.__init__(self,
                           info = {
                                    "input_images": "images", 
                                    "last_layer" : "fc8/BiasAdd",
                                    "file_name" : "ensemble",
                                    "url" : "ensamble",
                                    "model_name" : "ensemble"   
                                    },
                           dataset = dataset  
                          )
        self.ckpts = []
        self._models = []
        self._multiple_inputs = multiple_inputs
        self._type = type
        self._layers = []

    def add_models(self, models):
        if not type(models) == list:
            models = [models]
            
        for m in models:
            if not isinstance(m, BaseModel):
                raise Exception("The model must be a ""BaseModel"" type rather than {}".format(type(m)))
        
        self._models += models

                
    def _create_cnn(self, inputs, batch_norm = False, pretrained_weights = None): 
                if not self._models:
                    raise Exception("There is any model added.")
                cnn_layers = []
                if self._multiple_inputs:
                    
                    channels = self.hparams.channels // len(self._models)
                for i, model in enumerate(self._models):  
                    input_images = inputs if not self._multiple_inputs else  inputs[...,i*channels: (i+1)*channels]
                    
                    
                    hparams = Hyperparameters()
                    hparams.cut_layer = "" if model.cut_layer is None else model.cut_layer
                    hparams.fine_tunning = True
                    model.hparams = hparams
                    model._build_only_cnn(input_images, self.dict_model["istraining"])
                    cnn_layers.append(model.get_layers(-1))
                    self._layers += model._layers 
                
                if self._type == EnsembleType.DECAF:
                    self.add(layers.Decaf, cnn_layers)
                elif self._type == EnsembleType.CONV3D:
                   
                    cnns_nomalized = [ tf.nn.lrn(cnn.output) for cnn in cnn_layers] #local_response_normalization
                    cnns_nomalized = [tf.expand_dims(cnn, axis = 1) for cnn in cnns_nomalized]
                    input = tf.concat( cnns_nomalized, axis = 1, name='3D_concat')
                    self.add(layers.Conv3d, input = input, temp_filter_size = len(cnn_layers), 
                             num_filters = input.shape.as_list()[-1], kernel = [1,1])
                    self._layers[-1].output = tf.squeeze(self._layers[-1].output,axis = 1)
                    




    def _create_fully_connected(self, cnn_output, batch_norm = None):
        self.add(layers.FullyConnected, hidden_units = 4096, keep = 0.3, name = "fc1", istraining = self._istraining)
        self.add(layers.FullyConnected, hidden_units = 4096, keep = 0.3, name = "fc2", istraining = self._istraining)
        self.add(layers.Logits, num_classes = 1000, name = "logits")
        
    
    def build(self, input_images = None, labels = None, is_tf = True, new_fc = True, istraining = False):
            self.close()
            tf.reset_default_graph()

            self._istraining = istraining
   
            if  not new_fc:
                self.hparams.cut_layer = self.info["last_layer"]
                self.hparams.fine_tunning = True
                
            with tf.device("cpu:0"):
                if self.dataset.istfrecord():
                    input_data, labels = self.dataset.get_tfrecord_data()
                    self.hparams.bottleneck = False
                    
                else:
                    h, w = self.hparams.height, self.hparams.width
                    if self.hparams.data_augmentation and not self.hparams.auto_crop == (0,0): 
                        
                        h = self.hparams.height - self.hparams.auto_crop[0]
                        w = self.hparams.width - self.hparams.auto_crop[1]
                    input_data = tf.placeholder(tf.float32, shape=[None,h, w, self.hparams.channels], name='images')
                    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
            
                self.dict_model["images"] = input_data 
                self.dict_model["labels"] = labels
            self.dict_model["keep"] = tf.placeholder(tf.float32, name='dropout_keep')
            self.dict_model["istraining"] = tf.placeholder(tf.bool, name='istraining')
             
            self._transfer_learning(input_data, only_cnn = new_fc) if  is_tf else self._from_scratch(
                input_data, only_cnn = new_fc)
            outs = []
            for i, model in enumerate(self._models):
                    input_images = self.dict_model["images"] if not self._multiple_inputs else self.dict_model["images"][:,:,:,i*3:(i+1)*3]
                    model._build_only_cnn(input_images, istraining, is_tf = is_tf)
                    dict_weights = np.load(os.path.join(model._ckpt_dir, model.info["weights"]))
                    for l in model._layers:
                        l.ckpt_weigths = dict_weights
                        l.build()
                        l.model_name = self.info["model_name"]
                        l._model_input_data_placeholder = self.dict_model["images"]       
                        l._labels_placeholder = self.dict_model["labels"]
                        l._keep_placeholder= self.dict_model["keep"]
                        l._istraining_placeholder = self.dict_model["istraining"]
                    
                    outs.append(model.get_output().output)
            
  
            ensamble = tf.stack(outs,-1)
            self.add(layers.EnsambleMean, ensamble)
            self.add(layers.Flatten) 
            if  new_fc:
                self.add_fully_connected()
            
#             if istraining:
#                 self.generate_optimization_parameters()
                
        
#             if self.sess is None:
#                 self.sess, self._saver = self.create_monitored_session(self.dict_model)
            
            for model in self._models:
                for  l in model._layers:
                    l._sess = self.sess
                    l._logits = self._layers[-1]
                    
#             dict_weights = np.load(os.path.join(self._ckpt_dir, self.info["weights"]))
#             for l in self._layers: print("***** ",l)
            
            for l in self._layers:
                if l.ckpt_weigths is None:
#                     print("***** ",l)
                    l.ckpt_weigths = dict_weights
                    l.build()
                    l.model_name = self.info["model_name"]
                    l._model_input_data_placeholder = self.dict_model["images"]       
                    l._labels_placeholder = self.dict_model["labels"]
                    l._keep_placeholder= self.dict_model["keep"]
                    l._istraining_placeholder = self.dict_model["istraining"]
                l._sess = self.sess
                l._logits = self._layers[-1]
       

            self.built = True
    
    
    
    def _transfer_learning(self, inputs, only_cnn = False):
        self._create_cnn(inputs, self.hparams.batch_norm)
        
        if not self.hparams.fine_tunning:
            print("Stopping gradient")
            self._layers[-1].stop_gradient() #Just use it in case of transfer learning without fine tunning
        
        if  not only_cnn: 
            self._create_fully_connected(self.hparams.normalization)
              

   
        
        
    
    

