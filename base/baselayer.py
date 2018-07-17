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
    
    def __init__(self, input, name = "conv/conv", type = "conv", istraining = None):
        self.input = input
        self.output = None
        self.weights = None
        self.biases = None
        self.kernel = None
        self.stride = None
        self.name = name
        self.type = type
        self._my_input_data_placeholder = None
        self._model_input_data_placeholder = None
        self._labels_placeholder = False
        self._keep_placeholder= None
        
        self._logits = None
        self._sess = None
        self._model_name = None
        self.istraining = istraining
        self._build()
        self.shape = self.output.shape
        


    
    @abstractmethod
    def _build(self): pass
    
    def stop_gradient(self):
        self.output = tf.stop_gradient(self.output)
    
    def __str__(self):
        return str(self.name)
    
    def __delete__(self, instance):
        del self.input
        del self.output
        del self.weights
        del self.biases
        del self._input_data_placeholder 
        del self._labels_placeholder
        del self._keep_placeholder
        del self._istraining_placeholder
        del self._logits 

        
        del self.value
    def get_model_nome(self):
        return self._model_name
    
    def summary(self):
        params = 0 if self.weights is None else \
            (
            np.prod(self.weights.shape.as_list()[1:]) * \
            (1 if not self.type=="fc" else np.prod(self.input.shape.as_list()[1:]) ) 
            ) +\
            np.prod(self.biases.shape.as_list()[1:])
            
        input_value = self.input.shape.as_list()[1:] if not type(self.input)==list else tuple([input.shape.as_list()[1:] for input in self.input])
            
        return [ self._model_name, self.name, self.type, input_value, self.output.shape.as_list()[1:], self.kernel if  self.kernel else "-", self.stride if  self.stride else "-", params ]
   
    
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

    def get_3d_filters(self, temp_kernel_size, channels, spatial_kernel, out_filters, name = "conv"):
        xavier_init = tf.contrib.layers.xavier_initializer()
        filter = tf.Variable(xavier_init([temp_kernel_size]+ spatial_kernel+ [channels, out_filters]),
                             dtype=tf.float32, name="{}/weights".format(self.name))

        biases = tf.Variable(xavier_init([out_filters]), 
                           name="{}/biases".format(self.name))
        
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
            fc = self.fc_cosnorm(input, w,b, name = "{}/cosNorm".format(self.name))
        elif batch_norm == "layer_norm":
            fc = tf.nn.bias_add(fc, b)
            fc = tf.contrib.layers.layer_norm(fc, scope = "{}/layer_norm".format(name))
        elif batch_norm == "batch_norm":
            fc = tf.matmul(input, w)
            fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(self.name))
            fc = self.batch_norm(fc, name = "{}/batchNorm".format(self.name))

        else:
            fc = tf.matmul(input, w)
            fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(self.name))

        if keep is not None:
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
    
    def conv3d(self, tensor, temp_kernel_size, spatial_kernel, num_filters, stride=[1,1,1,1,1], batch_norm = None, name = "conv3d"):   
        channels = int(tensor.shape[4])
        filter, bias = self.get_3d_filters(temp_kernel_size, channels, spatial_kernel, num_filters, name = name)
        
        temp_layer = tf.nn.conv3d(tensor, filter, stride, data_format= "NDHWC",
                                  padding='VALID', name=name)
        temp_layer = tf.nn.bias_add(temp_layer, bias)
        if batch_norm == "layer_norm":
                fc = tf.contrib.layers.layer_norm(fc, scope = "{}/layer_norm".format(name))
        elif batch_norm is not None:
            fc = self.batch_norm(fc, "{}/batch_norm".format(name) )

        temp_layer = tf.nn.relu(temp_layer)
        self.output = temp_layer
        return temp_layer


    def conv2d(self, x, out_channels, kernel,  name = "conv",  batch_norm = False, padding = "SAME"):
        depth = x.shape.as_list()[-1]
        w, b = self.get_weights(depth, out_channels, kernel)
        self.stride = [1,1]
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], data_format= "NHWC", 
                         padding = padding, name = self.name)
        if batch_norm == "layer_norm":
            conv = tf.nn.bias_add(conv, b)
            conv = tf.contrib.layers.layer_norm(conv, scope = "{}/layer_norm".format(name))
        elif batch_norm == "cos_norm":
            conv = self.conv2d_cosnorm(conv, x, w, b, name = "{}/cos_norm".format(name))
        elif batch_norm is not None:
            conv = tf.nn.bias_add(conv, b)
            conv = self.batch_norm(conv, "{}/batch_norm".format(name) )
      
        conv = tf.nn.relu(conv)
        
        self.output = conv
        return conv
    
    def conv2d_fc(self, x, out_channels, kernel,  name = "conv",  activation = None, keep = None, batch_norm = None, padding = "VALID"):
        depth = x.shape.as_list()[-1]
        w, b = self.get_weights(depth, out_channels, kernel)
        self.stride = [1,1]
       
        fc = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], data_format= "NHWC", 
                         padding = padding, name = self.name)
        
        if batch_norm == "layer_norm":
            fc = tf.nn.bias_add(fc, b)
            fc = tf.contrib.layers.layer_norm(fc, scope = "{}/layer_norm".format(name))
        elif batch_norm == "cos_norm":
            fc = self.conv2d_cosnorm(fc, x, w, b, name = "{}/cos_norm".format(name))
        elif batch_norm is not None:
            fc = tf.nn.bias_add(fc, b)
            fc = self.batch_norm(fc, "{}/batch_norm".format(name) )
        
        if keep is not None:
            fc = tf.nn.dropout(fc, keep, name="{}/dropout".format(self.name))


        if activation == "sigmoid":
            fc = tf.nn.sigmoid(fc, name="{}/sigmoid".format(self.name))
        elif activation == "tanh":
            fc = tf.nn.tanh(fc, name="{}/tanh".format(self.name))
        else:
            fc = tf.nn.relu(fc, name="{}/relu".format(self.name))
            
        self.output = fc
        
        return fc


    def batch_norm(self, tensor, name = None):
        return  tf.layers.batch_normalization(tensor, 
                                               fused=True, 
#                                                name = name,
                                               training =  self._istraining_placeholder)
        
    def conv2d_cosnorm(self, conv, x, w, biases, bias=0.00001, name = "cosNorm"):

        kernel_sum = tf.reduce_sum(tf.square(w),[0,1,2])
        kernel_biases = tf.add(kernel_sum,tf.square(biases))
        kernel_norm = tf.sqrt(kernel_biases)
        x_sum = tf.nn.conv2d(tf.square(x), tf.ones_like(w), [1, 1, 1, 1], padding='SAME')
        x_biases = tf.add(x_sum,tf.square(bias))
        x_norm = tf.sqrt(x_biases) 
        biases_mul = tf.multiply(biases,bias)
        bias = tf.nn.bias_add(conv, biases_mul)
        conv_norm_kernel = tf.div(bias,kernel_norm)
        conv_norm = tf.div(conv_norm_kernel,x_norm)
        return conv_norm

    def fc_cosnorm(self, x, w, biases, bias=0.00001, name = "cosNorm"):
        x = tf.add(x, bias)
        w = tf.add(w, bias)

        y = tf.matmul(x, w) + biases

        x = tf.reduce_sum(tf.square(x),1, keepdims=True)
        x = tf.sqrt(x)

        w = tf.reduce_sum(tf.square(w),0, keepdims=True)
        w = tf.sqrt(w)


        return tf.divide(y ,(x * w), name = name)
    
    def visualize_weights(self):
        return utils.weights_visualization(self._sess, self.weights)
    
    
    
    def visualize_features(self, features= [1], activation = False):
        if type(features) == int:
            features = [features]
        elif not type(features) == list:
            raise Exception("Features needs to be a list or an integer")
            
        if activation or  self.weights is None:
            name = self.output.name
        else:
            name = self.weights.name
            name = "/".join(name.split("/")[:-1])+":0"
            
        images = []
        
        for feature in features:
            image= utils.filters_visualization(self._sess, self._model_input_data_placeholder, self._istraining_placeholder, name, feature)
            images.append(image)
        return images
    
    def activation_maps(self, images=[], labels=[]):
        
#         images = self.dataset.get_sample() if not images or len(images) == 0 else images
        if not type(images) == list:
            images = [images]
        
        if not type(labels) == list:
            labels = [labels]
        elif not labels:
            labels = [-1]*len(images)
            
        image_results = []
        for i, image in enumerate(images):
            image = np.expand_dims(image, axis=0) if len(list(image.shape)) == 3 else image
            results = utils.grad_CAM_plus(self._sess, image, self._model_input_data_placeholder, 
                                           self._logits.output, self._labels_placeholder, self._keep_placeholder, self.output, labels[i], self.name)
            image_results.append(results)
        return image_results
            
            
        
        
        

    
    

