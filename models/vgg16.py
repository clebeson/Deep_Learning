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


class Vgg16(BaseModel):
    def __init__(self, dataset, cut_layer = None):

        BaseModel.__init__(self,
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "fc8/BiasAdd",
                                    "file_name" : "vgg_16.ckpt",
                                    "url" : "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                                    "model_name" : "vgg_16"   
                                    },
                       dataset = dataset, cut_layer = cut_layer
                      )
        

    
#     def _get_cnn(self, inputs, batch_norm = None, pretrained_weights = None): 
#         net = self.repeat(2, bf.conv2d, inputs, 64, [3, 3], "conv1",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool1")
#         if self.hparams.cut_layer == "pool1": return net
        
#         net = self.repeat(2, bf.conv2d, net, 128, [3, 3], "conv2",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool2")
#         if self.hparams.cut_layer == "pool2": return net
        
#         net = self.repeat(3, bf.conv2d, net, 256, [3, 3], "conv3",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool3")
#         if self.hparams.cut_layer == "pool3": return net

#         net = self.repeat(3, bf.conv2d, net, 512, [3, 3], "conv4",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool4")
#         if self.hparams.cut_layer == "pool4": return net

#         net = self.repeat(3, bf.conv2d, net, 512, [3, 3], "conv5",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool5")
        
#         return net
    
#     def _get_fully_connected(self, cnn_output, batch_norm = None):
#         fc = bf.fully_connected( input = cnn_output, hidden_units = 4096, name = "fc6")
#         fc = bf.fully_connected( input = fc, hidden_units = 4096, name = "fc7")
#         logits = bf.fully_connected( input = fc, hidden_units = 1000, name = "fc8", activation = None, batch_norm = False)
#         return logits


    def _create_cnn(self, inputs, batch_norm = None, pretrained_weights = None): 
        self.add_repeated(2, layers.Conv2d, 64, [3, 3], name = "conv1",  norm = False, istraining = self._istraining)
        self.add(layers.MaxPool,name = "pool1")
        if self.hparams.cut_layer == "pool1": return
        
        self.add_repeated(2,layers.Conv2d,  128, [3, 3], name = "conv2",  norm = False, istraining = self._istraining)
        self.add(layers.MaxPool, name = "pool2")
        if self.hparams.cut_layer == "pool2": return
        
        self.add_repeated(3,layers.Conv2d,  256, [3, 3], name = "conv3",  norm = False, istraining = self._istraining)
        self.add(layers.MaxPool, name = "pool3")
        if self.hparams.cut_layer == "pool3": return

        self.add_repeated(3, layers.Conv2d,  512, [3, 3], name = "conv4",  norm = False, istraining = self._istraining)
        self.add(layers.MaxPool, name = "pool4")
        if self.hparams.cut_layer == "pool4": return

        self.add_repeated(3,layers.Conv2d,  512, [3, 3], name = "conv5",  norm = False, istraining = self._istraining)
        self.add(layers.MaxPool, name = "pool1")

    
    def _create_fully_connected(self, batch_norm = None):
        self.add(layers.FullyConnected, hidden_units = 4096, keep = 0.3, name = "fc6", istraining = self._istraining)
        self.add(layers.FullyConnected, hidden_units = 4096, keep = 0.3, name = "fc7", istraining = self._istraining)
        self.add(layers.Logits, num_classes = 1000, name = "fc8")
        
        
        
        
