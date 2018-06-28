from abc import ABCMeta, abstractmethod 
import pickle
import numpy as np
import os
import sys
import tensorflow as tf
import numpy as np
import os.path
import utils
import math
from time import time
from random import shuffle
from itertools import izip as zip



class BaseLayer():
    
    __metaclass__ = ABCMeta
    
    def __init__(self, input, name = "conv/conv", type = "conv"):
        self.input = input
        self.output = None
        self.weights = None
        self.biases = None
        self.kernel = None
        self.stride = None
        self.name = name
        self.type = type
        self._build()
        self.shape = self.output.shape
    
    @abstractmethod
    def _build(self): pass
    
    def stop_gradient(self):
        self.output = tf.stop_gradient(self.output)
    
    def __str__(self):
        return str(self.name)
    
    def summary(self):
        params = 0 if self.weights is None else \
            (
            np.prod(self.weights.shape.as_list()[1:]) * \
            (1 if not self.type=="fc" else np.prod(self.input.shape.as_list()[1:]) ) 
            ) +\
            np.prod(self.biases.shape.as_list()[1:])
        return [ self.name, self.type, self.input.shape.as_list()[1:], self.output.shape.as_list()[1:], self.kernel if  self.kernel else "-", self.stride if  self.stride else "-", params ]
    def shape(self):
        return self.output.shape
    def conv3d(self, tensor, temp_kernel, space_kernel, num_filters, stride=[1,1,1,1,1], name = "conv3d"):   
        channels = int(tensor.shape[4])
        filter, _ = self.get_3d_filters(temp_kernel, channels, num_filters, 
                                        space_kernel, name)

        temp_layer = tf.nn.conv3d(tensor, filter, stride, data_format= "NDHWC",
                                  padding='VALID', name=name)
        # temp_layer = cos.conv3d_cosnorm(input, filters, strides=[1,1,1,1,1], padding='VALID')
        temp_layer =  self.batch_norm(temp_layer)
        #temp_layer = tf.nn.bias_add(temp_layer, bias)


        temp_layer = tf.nn.relu(temp_layer)
        self.output = temp_layer
        return temp_layer

    def get_3d_filters(self, temporal_kernel_size, channels, temp_filters_size, spacial_kernel_size=1, id = 0):
        filter = tf.Variable(tf.random_normal([ temporal_kernel_size, 
                                               spacial_kernel_size, 
                                               spacial_kernel_size,   
                                               channels, 
                                               temp_filters_size ]),
                             dtype=tf.float32, name="3d_filter")

        biases = tf.Variable(tf.random_normal([temp_filters_size]), 
                           name="B_temp")
        
        self.weights = filter
        self.biases = biases

        
        return filter, biases

    def get_weights(self, w_inputs, w_output, kernel_size = []):
        xavier_init = tf.contrib.layers.xavier_initializer()
        weights= tf.Variable(xavier_init(kernel_size + [w_inputs, w_output]),
                             name="{}/weights".format(self.name))
        biases =  tf.Variable(xavier_init([w_output]), 
                              name="{}/biases".format(self.name))
        self.weights = weights
        self.biases = biases
        self.kernel = kernel_size
        return weights, biases



    def fully_connected(self, input, hidden_units, keep = None, activation = "relu", name = "fc", batch_norm = None):

        input_list = input.shape.as_list()
        input_size = np.prod(input_list[1:])
        w, b = self.get_weights(input_size , hidden_units)
        if batch_norm == "cos_norm":
            fc = fc_cosnorm(input, w, name = "{}/cosNorm".format(self.name))

        elif batch_norm == "batch_norm":
            fc = tf.matmul(input, w, b)
            fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(self.name))
            fc = batch_norm(fc, name = "{}/batchNorm".format(layer_id))

        else:
            fc = tf.matmul(input, w)
            fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(self.name))

        if keep:
    #             print "Dropout activated in layer {}".format(name)
            fc = tf.nn.dropout(fc, keep, name="{}/dropout".format(self.name))

        if activation == "relu":
            fc = tf.nn.relu(fc, name="{}/relu".format(self.name))
        elif activation == "sigmoid":
            fc = tf.nn.sigmoid(fc, name="{}/sigmoid".format(self.name))
        self.output = fc
        return fc


    def avgpool(self, tensor, k=2, d=1):
        if len(tensor.shape) == 5:
            return tf.nn.avg_pool3d(tensor, ksize=[1,  d,k, k, 1], 
                                    strides=[1, d , k, k, 1], 
                                    data_format= "NDHWC" , padding='VALID')

        return tf.nn.avg_pool(ksize=[1, k, k, d], 
                              strides=[1, k, k, d], padding='VALID' )

    def maxpool(self, tensor, k=2, d=1, stride = 1, name = "pool"):
        self.kernel = [k,k]
        self.stride = [k,k]
        if len(tensor.shape) == 5:
            return tf.nn.max_pool3d(tensor, ksize=[1,  d,k, k, 1], 
                                    strides=[1, d , stride, stride, 1], 
                                    data_format= "NDHWC" , padding='VALID', name = name)

        return tf.nn.max_pool(tensor, ksize=[1, k, k, d], strides=[1, stride, stride, d],
                              padding='VALID', name = name )

    def tensor_maxpool(self, tensor, axis = 4, layer_id = 0):
        if len(tensor.shape) == 4:
            tensor=tf.expand_dims(tensor,1)
        tensor_mp = tf.reduce_max(tensor, axis=axis, keepdims = True)
        return tensor_mp

    def tensor_sum(self, tensor, axis = 4, layer_id = 0):
        tensor_sum = tf.reduce_sum(tensor, axis=axis, keepdims = True)
        # tensor_mp =  batch_norm(tensor_mp, scope = "BN_temp/{}".format(layer_id))
        return tensor_sum

    def tensor_avgpool(self, tensor, axis = 4, layer_id = 0):
        tensor_ap = tf.reduce_mean(tensor, axis=axis)

        # tensor_ap =  batch_norm(tensor_ap, scope = "BN_space/{}".format(layer_id))
        return tensor_ap        

    def conv2d(self, x, out_channels, kernel,  name = "conv",  batch_norm = False, padding = "SAME"):
        depth = x.shape.as_list()[-1]
        w, b = self.get_weights(depth, out_channels, kernel)
        self.stride = [1,1]
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], data_format= "NHWC", 
                         padding = padding, name = self.name)
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        if batch_norm:
            conv = self.batch_norm(conv, "{}/batch_norm" % name)
        self.output = conv
        return conv
    
    def conv2d_fc(self, x, out_channels, kernel,  name = "conv",  activation = None, keep = None, batch_norm = None, padding = "VALID"):
        depth = x.shape.as_list()[-1]
        w, b = self.get_weights(depth, out_channels, kernel)
        self.stride = [1,1]
       
        fc = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], data_format= "NHWC", 
                         padding = padding, name = self.name)
        fc = tf.nn.bias_add(fc, b)
        if batch_norm:
            fc = self.batch_norm(fc, "{}/batch_norm" % name)
        
        if keep:
            fc = tf.nn.dropout(fc, keep, name="{}/dropout".format(self.name))

        if activation == "relu":
            fc = tf.nn.relu(fc, name="{}/relu".format(self.name))
        elif activation == "sigmoid":
            fc = tf.nn.sigmoid(fc, name="{}/sigmoid".format(self.name))
        self.output = fc
        
        return fc


    def batch_norm(self, tensor, name = None):
         return  tf.layers.batch_normalization(tensor, fused=True, 
                                               data_format='NCHW', 
                                               name = name,
                                               training =  self.param.is_training)

    def fc_cosnorm(self, x, w, biases, bias=0.00001, name = "cosNorm"):
        x = tf.add(x, bias)
        w = tf.add(w, bias)

        y = tf.matmul(x, w) + biases

        x = tf.reduce_sum(tf.square(x),1, keepdims=True)
        x = tf.sqrt(x)

        w = tf.reduce_sum(tf.square(w),0, keepdims=True)
        w = tf.sqrt(w)


        return tf.divide(y ,(x * w), name = name)
        
        

    
    

