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


class Lenet(BaseModel):
    def __init__(self,hparams):
        BaseModel.__init__(self, hparams, 
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "fc8/BiasAdd",
                                    "file_name" : "vgg_16.ckpt",
                                    "url" : "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                                    "model_name" : "lenet"   
                                    }
                      )


    def _get_cnn(self, inputs, batch_norm = None):
        net = bf.conv2d(inputs, 32, [5, 5], name='conv1', batch_norm = batch_norm)
        net = bf.max_pool2d(net, [2, 2], 2, name='pool1',  batch_norm = batch_norm)
        net = bf.conv2d(net, 64, [5, 5], name='conv2',  batch_norm = batch_norm)
        net = bf.max_pool2d(net, [2, 2], 2, name='pool2',  batch_norm = batch_norm)
        return net
    


        
        
