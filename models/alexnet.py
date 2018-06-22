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


class Alexnet(BaseModel):
    def __init__(self,hparams):
        BaseModel.__init__(self, hparams, 
                   info = {
                                "input_images": "images", 
                                "last_layer" : "fc8/BiasAdd",
                                "file_name" : "vgg_16.ckpt",
                                "url" : "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                                "model_name" : "alexnet"   
                                }
                  )


    def _get_cnn(self, inputs, batch_norm = False):
        net = bf.conv2d(inputs, 64, [11, 11], 4, padding='VALID', name='conv1')
        net = bf.maxpool(net, [3, 3], [2,2], name='pool1')
        net = bf.conv2d(net, 192, [5, 5], name='conv2')
        net = bf.maxpool(net, [3, 3], 2, name='pool2')
        net = bf.conv2d(net, 384, [3, 3], name='conv3')
        net = bf.conv2d(net, 384, [3, 3], name='conv4')
        net = bf.conv2d(net, 256, [3, 3], name='conv5')
        net = bf.maxpool(net, [3, 3], 2, name='pool5')
        return net


        
        
