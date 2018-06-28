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


    
def conv3d(tensor, temp_kernel, space_kernel, num_filters, stride=[1,1,1,1,1], name = "conv3d"):   

    channels = int(tensor.shape[4])
    filter, _ = self.get_3d_filters(temp_kernel, channels, num_filters, 
                                    space_kernel, name)

    temp_layer = tf.nn.conv3d(tensor, filter, stride, data_format= "NDHWC",
                              padding='VALID', name=name)
    # temp_layer = cos.conv3d_cosnorm(input, filters, strides=[1,1,1,1,1], padding='VALID')
    temp_layer =  self.batch_norm(temp_layer)
    #temp_layer = tf.nn.bias_add(temp_layer, bias)
        
        
    temp_layer = tf.nn.relu(temp_layer)
    return temp_layer

def get_3d_filters( temporal_kernel_size, channels, temp_filters_size, spacial_kernel_size=1, id = 0):
    filter = tf.Variable(tf.random_normal([ temporal_kernel_size, 
                                           spacial_kernel_size, 
                                           spacial_kernel_size,   
                                           channels, 
                                           temp_filters_size ]),
                         dtype=tf.float32, name="3d_filter")

    bias = tf.Variable(tf.random_normal([temp_filters_size]), 
                       name="B_temp")
    return filter, bias
    
def get_weights(w_inputs, w_output, kernel_size = [], name = "conv/conv"):
    xavier_init = tf.contrib.layers.xavier_initializer()
    weights= tf.Variable(xavier_init(kernel_size + [w_inputs, w_output]),
                         name="{}/weights".format(name))
    biases =  tf.Variable(xavier_init([w_output]), 
                          name="{}/biases".format(name))

    return weights, biases
    
    
    
def fully_connected(input, hidden_units, keep = None, activation = "relu", name = "fc", batch_norm = None):

    input_list = input.shape.as_list()
    input_size = np.prod(input_list[1:])
    w, b = get_weights(input_size , hidden_units, name = name)

    if batch_norm == "cos_norm":
        fc = fc_cosnorm(input, w, name = "{}/cosNorm")

    elif batch_norm == "batch_norm":
        fc = tf.matmul(input, w, b)
        fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(name))
        fc = batch_norm(fc, name = "{}/batchNorm".format(layer_id))

    else:
        fc = tf.matmul(input, w)
        fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(name))

    if keep:
#             print "Dropout activated in layer {}".format(name)
        fc = tf.nn.dropout(fc, keep, name="{}/dropout".format(name))

    if activation == "relu":
        fc = tf.nn.relu(fc, name="{}/relu".format(name))
    elif activation == "sigmoid":
        fc = tf.nn.sigmoid(fc, name="{}/sigmoid".format(name))

    return fc

def logits_layer(fc_layer):
    out_shape = fc_layer.shape.as_list()
    out_size = np.prod(out_shape[1:])
    w, b = self.get_weights(out_size, self.hparams.num_classes, 
                            name = "logits/weight")
    logits = tf.add(tf.matmul(fc_layer, w), b, name="logits")
    return logits

def avgpool(tensor, k=2, d=1):
    if len(tensor.shape) == 5:
        return tf.nn.avg_pool3d(tensor, ksize=[1,  d,k, k, 1], 
                                strides=[1, d , k, k, 1], 
                                data_format= "NDHWC" , padding='VALID')

    return tf.nn.avg_pool(ksize=[1, k, k, d], 
                          strides=[1, k, k, d], padding='VALID' )
                
def maxpool(self, tensor, k=2, d=1, name = "pool"):
    if len(tensor.shape) == 5:
        return tf.nn.max_pool3d(tensor, ksize=[1,  d,k, k, 1], 
                                strides=[1, d , k, k, 1], 
                                data_format= "NDHWC" , padding='VALID', name = name)

    return tf.nn.max_pool(tensor, ksize=[1, k, k, d], strides=[1, k, k, d],
                          padding='VALID', name = name )
                                                    
def tensor_maxpool(tensor, axis = 4, layer_id = 0):
    if len(tensor.shape) == 4:
        tensor=tf.expand_dims(tensor,1)
    tensor_mp = tf.reduce_max(tensor, axis=axis, keepdims = True)
    return tensor_mp

def tensor_sum( tensor, axis = 4, layer_id = 0):
    tensor_sum = tf.reduce_sum(tensor, axis=axis, keepdims = True)
    # tensor_mp =  batch_norm(tensor_mp, scope = "BN_temp/{}".format(layer_id))
    return tensor_sum

def tensor_avgpool( tensor, axis = 4, layer_id = 0):
    tensor_ap = tf.reduce_mean(tensor, axis=axis)

    # tensor_ap =  batch_norm(tensor_ap, scope = "BN_space/{}".format(layer_id))
    return tensor_ap        
        
def conv2d(x, out_channels, kernel_size,  name = "conv",  batch_norm = False, padding = "SAME"):
    depth = x.shape.as_list()[-1]
    w, b = self.get_weights(depth, out_channels,kernel_size, name)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], data_format= "NHWC", 
                     padding = padding)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    if batch_norm:
        x = self.batch_norm(x, "{}/batch_norm" % name)

    return x 

    
def batch_norm(self, tensor, name = None):
     return  tf.layers.batch_normalization(tensor, fused=True, 
                                           data_format='NCHW', 
                                           name = name,
                                           training =  self.param.is_training)

def fc_cosnorm(x, w, biases, bias=0.00001, name = "cosNorm"):
    x = tf.add(x, bias)
    w = tf.add(w, bias)

    y = tf.matmul(x, w) + biases

    x = tf.reduce_sum(tf.square(x),1, keepdims=True)
    x = tf.sqrt(x)

    w = tf.reduce_sum(tf.square(w),0, keepdims=True)
    w = tf.sqrt(w)


    return tf.divide(y ,(x * w), name = name)

def regularize(loss, type = 1, scale = 0.005, scope = None):
    if type == 1:
        regularizer = tf.contrib.layers.l1_regularizer( scale=scale,
                                                       scope=scope)
    else:
        regularizer = tf.contrib.layers.l2_regularizer( scale=scale, 
                                                       scope=scope)

    weights = tf.trainable_variables() # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
    regularized_loss = loss + regularization_penalty
    return regularized_loss     


    
    

