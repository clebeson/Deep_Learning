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
        

    
    def _create_cnn(self, inputs, batch_norm = None, pretrained_weights = None): 
        self.add_repeated(2, layers.Conv2d, 64, [3, 3], istraining = self.dict_model["istraining"], name = "conv1",  norm = False)
        self.add(layers.MaxPool,name = "pool1")
        if self.hparams.cut_layer == "pool1": return
        
        self.add_repeated(2,layers.Conv2d,  128, [3, 3], istraining = self.dict_model["istraining"],name = "conv2",  norm = False)
        self.add(layers.MaxPool, name = "pool2")
        if self.hparams.cut_layer == "pool2": return
        
        self.add_repeated(3,layers.Conv2d,  256, [3, 3], istraining = self.dict_model["istraining"],name = "conv3",  norm = False)
        self.add(layers.MaxPool, name = "pool3")
        if self.hparams.cut_layer == "pool3": return

        self.add_repeated(3, layers.Conv2d,  512, [3, 3], istraining = self.dict_model["istraining"],name = "conv4",  norm = False)
        self.add(layers.MaxPool, name = "pool4")
        if self.hparams.cut_layer == "pool4": return

        self.add_repeated(3,layers.Conv2d,  512, [3, 3], istraining = self.dict_model["istraining"],name = "conv5",  norm = False)
        self.add(layers.MaxPool, name = "pool5")

#     def _create_cnn(self, inputs, batch_norm = None, pretrained_weights = None): 
#         self.add(layers.Conv2d, 64, [3, 3],  istraining = self.dict_model["istraining"], name = "conv1/conv1_1",  norm = False)
#         self.add(layers.Conv2d, 64, [3, 3],  stride = [1,2,2,1], istraining = self.dict_model["istraining"], name = "conv1/conv1_2",  norm = False)
# #         self.add(layers.MaxPool,name = "pool1")
#         if self.hparams.cut_layer == "pool1": return
        
#         self.add(layers.Conv2d,  128, [3, 3], istraining = self.dict_model["istraining"],name = "conv2/conv2_1",  norm = False)
#         self.add(layers.Conv2d,  128, [3, 3], stride = [1,2,2,1], istraining = self.dict_model["istraining"],name = "conv2/conv2_2",  norm = False)
# #         self.add(layers.MaxPool, name = "pool2")
#         if self.hparams.cut_layer == "pool2": return
        
#         self.add_repeated(2,layers.Conv2d,  256, [3, 3], istraining = self.dict_model["istraining"],name = "conv3",  norm = False)
#         self.add(layers.Conv2d,  256, [3, 3], stride = [1,2,2,1], istraining = self.dict_model["istraining"],name = "conv3/conv3_3",  norm = False)
# #         self.add(layers.MaxPool, name = "pool3")
#         if self.hparams.cut_layer == "pool3": return

#         self.add_repeated(2, layers.Conv2d,  512, [3, 3], istraining = self.dict_model["istraining"],name = "conv4",  norm = False)
#         self.add(layers.Conv2d,  512, [3, 3], stride = [1,2,2,1], istraining = self.dict_model["istraining"],name = "conv4/conv4_3",  norm = False)

# #         self.add(layers.MaxPool, name = "pool4")
#         if self.hparams.cut_layer == "pool4": return

#         self.add_repeated(3,layers.Conv2d,  512, [3, 3], istraining = self.dict_model["istraining"],name = "conv5",  norm = False)
#         self.add(layers.Conv2d,  512, [3, 3], stride = [1,2,2,1], istraining = self.dict_model["istraining"],name = "conv5/conv5_3",  norm = False)
# #         self.add(layers.MaxPool, name = "pool5")

    
    def _create_fully_connected(self, batch_norm = None):
        self.add_repeated(2, layers.Conv2d, 64, [3, 3], name = "conv1",  norm = False, istraining = self.dict_model["istraining"])
        self.add(layers.FullyConnected, hidden_units = 4096, keep = 0.3, name = "fc6", istraining = self.dict_model["istraining"])
        self.add(layers.FullyConnected, hidden_units = 4096, keep = 0.3, name = "fc7", istraining = self.dict_model["istraining"])
        self.add(layers.Logits, num_classes = 1000, name = "fc8")
        
        
        
        
