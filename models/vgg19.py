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




class Vgg19(BaseModel): pass
#     def __init__(self,hparams, dataset):

#         BaseModel.__init__(self, hparams, 
#                        info = {
#                                     "input_images": "images", 
#                                     "last_layer" : "fc8/BiasAdd",
#                                     "file_name" : "vgg_19.ckpt",
#                                     "url" : "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz",
#                                     "model_name" : "vgg_19"   
#                                     },
#                            dataset = dataset
#                       )


#     def _get_cnn(self, inputs, batch_norm = False, pretrained_weights = None): 

#         net = self.repeat(2, bf.conv2d, inputs, 64, [3, 3], "conv1",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool1")
#         if self.hparams.cut_layer == "pool1": return net

#         net = self.repeat(2, bf.conv2d, net, 128, [3, 3], "conv2",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool2")
#         if self.hparams.cut_layer == "pool2": return net


#         net = self.repeat(4, bf.conv2d, net, 256, [3, 3], "conv3",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool3")
#         if self.hparams.cut_layer == "pool3": return net

        
#         net = self.repeat(4, bf.conv2d, net, 512, [3, 3], "conv4",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool4")
#         if self.hparams.cut_layer == "pool4": return net

#         net = self.repeat(4, bf.conv2d, net, 512, [3, 3], "conv5",  batch_norm)
#         net = bf.maxpool(tensor = net, name = "pool5")
#         return net

#     def _get_fully_connected(self, cnn_output, batch_norm = None):
#         fc = bf.fully_connected( input = cnn_output, hidden_units = 4096, name = "fc6")
#         fc = bf.fully_connected( input = fc, hidden_units = 4096, name = "fc7")
#         logits = bf.fully_connected( input = fc, hidden_units = 1000, name = "fc8", activation = None, batch_norm = False)
#         return logits

        
        

        
        
        
