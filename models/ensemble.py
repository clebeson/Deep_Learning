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
                    raise Eception("There is any model added.")
                cnn_layers = []
                if self._multiple_inputs:
                    channels = self.hparams.channels // len(self._models)
                    
                for i, model in enumerate(self._models):  
                    input_images = inputs if not self._multiple_inputs else  inputs[:,:,:,i*channels: (i+1)*channels]
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
        
    
    def build(self, is_tf = True, new_fc = True, istraining = False):
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
                    labels = tf.placeholder(tf.float32, shape=[None, self.hparams.num_classes], name='labels')
            
                self.dict_model["images"] = input_data 
                self.dict_model["labels"] = labels
            self.dict_model["keep"] = tf.placeholder(tf.float32, name='dropout_keep')
            self.dict_model["istraining"] = tf.placeholder(tf.bool, name='istraining')
             
            self._transfer_learning(input_data, only_cnn = new_fc) if  is_tf else self._from_scratch(
                input_data, only_cnn = new_fc)
                
            
            if  new_fc:
                self.add_fully_connected()
            
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
                if l._model_name is None:
                    l._model_name = "ensemble"
                    
            
            
            self.built = True
    
    
    
    def _transfer_learning(self, inputs, only_cnn = False):
        self._create_cnn(inputs, self.hparams.batch_norm)
        
        if not self.hparams.fine_tunning:
            print("Stopping gradient")
            self._layers[-1].stop_gradient() #Just use it in case of transfer learning without fine tunning
        
        if  not only_cnn: 
            self._create_fully_connected(self.hparams.normalization)
              

    
    def create_monitored_session(self, model,iter_per_epoch = 1):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
 
        sess = tf.Session(config=config)
        all_vars = tf.global_variables()
        all_vars.extend(tf.local_variables())
        saver = tf.train.Saver( var_list = all_vars, max_to_keep=1)
#         sess.run(tf.variables_initializer(all_vars))
        
       
        ckpt_path = self._generate_checkpoint_path()
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            
        if len(os.listdir(ckpt_path))  > 0: 
            try:
                print "Restoring the entire model."
                saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            except Exception as e:
                
                print("It was not possible to restore the model. All variables will be initialized with their default values.")
                sess.run(tf.variables_initializer(all_vars))
                print(e)
            
             
           
        else:
            
            restored = []
            to_restore = []

            self._models.sort(key=lambda model: len(model._layers),reverse=True)
            for model in self._models:
                to_restore =  [var for var in all_vars if var.name in model.vars_to_restore ]
                res = tf.train.Saver(to_restore)

                try:

                    model_path = os.path.join(self._ckpt_dir , model.info["file_name"])
                    for var in model_path:
                        print(var.name)
                    print("************************")

                    print("Restoring variables from: {}".format(model_path))
                    res.restore(sess, model_path)

                except:
                    to_restore_names = ["/".join(var.name.split("/")[1:]) for var in to_restore]
                    for var in to_restore:
                        print(var.name)
                    print("************************")

                    alread_restored = [var 
                                     for var in restored 
                                     if var.name.split("/")[0]  == model.info["model_name"]
                                     and "/".join(var.name.split("/")[1:]) in  to_restore_names
                                    ]
                   
                    to_restore.sort(key=lambda var:var.name)
                    alread_restored.sort(key=lambda var:var.name)
                    
                    if len(to_restore) != len(alread_restored):
                        print("Some variables could not be restored - (Amount={})".format(len(to_restore) - len(alread_restored)))
                       

                    for to, alread in zip(to_restore,alread_restored):
                        sess.run(to.assign(alread))

                restored += to_restore
                       
            restored_names = [var.name for var in restored]        
            unrestored = [var for var in all_vars if var.name not in restored_names]                    
            print("Initializing only unrestored variables")
            sess.run(tf.variables_initializer(unrestored))
            

                  
        
        return sess, saver
    
        
        
    
    

