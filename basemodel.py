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


class BaseModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_cnn(self, inputs, batch_norm = False ):  pass
    
    
    def _get_fully_connected(self, cnn_output, batch_norm = False): pass
    
    
    
    
    
    def __init__(self, hparams, info, dataset):
        self.hparams = hparams
        self.dataset = dataset
        self.info = info
        self.sess = None
        self.built = False
        self.dict_model = {
                      "global_step": None,
                      "images": None,
                      "labels": None,    
                      "loss" : None,
                      "optimizer": None,
                      "accuracy": None,
                      "predictions": None,
                      "keep": None,
                      "bottleneck_tensor":None,
                      "bottleneck_input":None
                  }
        self.vars_to_restore = []
            
        
    def _build_only_cnn(self, input_images, is_tf = True):
            self.built = False
            print "Buiding cnn from ""{}"" model.".format(self.info["model_name"])
            self.dict_model["images"] = input_images
                
            with tf.name_scope(self.info["model_name"]):
                cnn  = self._transfer_learning(input_images, only_cnn = True) if  is_tf else self._from_scratch(input_images, 
                                                                                                                only_cnn = True)            
            self.built = True
            
#             g = tf.get_default_graph()
#             for i in g.get_operations():
#                 print(i.values())
            return cnn

   
        
    def build(self, is_tf = True, new_fc = True):
           
            self.built = False
            tf.reset_default_graph()
            
            if  not new_fc:
                self.hparams.cut_layer = self.info["last_layer"]
                self.hparams.fine_tunning = True
            with tf.device("cpu:0"):
                if self.dataset.istfrecord():
                    self._input_data, self._labels = sef.dataset.get_tfrecord_data()
                    self.hparams.bottleneck = False
#                     input_images =  self.input_data
                else:
                    self._input_data = tf.placeholder(tf.float32, shape=[None,self.hparams.height, 
                                                              self.hparams.width, 
                                                              self.hparams.channels], 
                                                              name='images')
                    
                    self._labels = tf.placeholder(tf.float32, shape=[None, self.hparams.num_classes], name='labels')
                    self.dict_model["images"] = self._input_data 
            dropout_keep  =  tf.placeholder(tf.float32, name='dropout_keep')
            self.dict_model["keep"] = dropout_keep


            
                
            with tf.name_scope(self.info["model_name"]):
               # Create a VGG16 model and reuse its weights.
                model  = self._transfer_learning(self._input_data, only_cnn = new_fc) if  is_tf else self._from_scratch(
                    self._input_data, only_cnn = new_fc)
                       
            if  not new_fc:
                logits = model         
            else:
                with tf.name_scope("flatten"):
                    flatten = tf.layers.flatten(model, name="flatten")

                logits = self.add_fully_connected(flatten)
      
            self.generate_optimization_parameters(logits)
            self.built = True
#             g = tf.get_default_graph()
#             for i in g.get_operations():
#                 print(i.values())
            
        
    def get_name(self):
        return self.info["model_name"]
    
    
    def conv3d(self, tensor, temp_kernel, space_kernel, num_filters, stride=[1,1,1,1,1], name = "conv3d"):   

        channels = int(tensor.shape[4])
        filter, _ = self.get_3d_filters(temp_kernel, channels, num_filters, space_kernel, name)
       
        temp_layer = tf.nn.conv3d(tensor, filter, stride, data_format= "NDHWC", padding='VALID', name=name)
        # temp_layer = cos.conv3d_cosnorm(input, filters, strides=[1,1,1,1,1], padding='VALID')
        temp_layer =  self.batch_norm(temp_layer)
        #temp_layer = tf.nn.bias_add(temp_layer, bias)
        
        
        temp_layer = tf.nn.relu(temp_layer)
        return temp_layer

    def get_3d_filters(self, temporal_kernel_size, channels, temp_filters_size, spacial_kernel_size=1, id = 0):
            filter = tf.Variable(tf.random_normal([ temporal_kernel_size, 
                                                   spacial_kernel_size, 
                                                   spacial_kernel_size,   
                                                   channels, 
                                                   temp_filters_size ]),
                                 dtype=tf.float32, name="3d_filter")
            
            bias = tf.Variable(tf.random_normal([temp_filters_size]), name="B_temp")
            return filter, bias
    
    def get_weights(self, w_inputs, w_output, kernel_size = [], name = "conv/conv"):
        xavier_init = tf.contrib.layers.xavier_initializer()
        weights= tf.Variable(xavier_init(kernel_size + [w_inputs, w_output]), name="{}/weights".format(name))
        biases =  tf.Variable(xavier_init([w_output]), name="{}/biases".format(name))

        return weights, biases
    
    
    
    def fully_connected(self, input, hidden_units, activation = "relu", name = "fc", batch_norm = None):
        
        input_list = input.shape.as_list()
        input_size = np.prod(input_list[1:])
        w, b = self.get_weights(input_size , hidden_units, name = name)
        
        batch_norm = self.hparams.batch_norm if batch_norm is None else batch_norm
        
        if batch_norm == "cos_norm":
            fc = self.fc_cosnorm(input, w, name = "{}/cosNorm")
    
        elif batch_norm == "batch_norm":
            fc = tf.matmul(input, w, b)
            fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(name))
            fc = self.batch_norm(fc, name = "{}/batchNorm".format(layer_id))
        
        else:
            fc = tf.matmul(input, w)
            fc = tf.nn.bias_add(fc, b, name="{}/biasesAdd".format(name))
        
        if self.hparams.keep and self.hparams.keep < 1.0:
#             print "Dropout activated in layer {}".format(name)
            fc = tf.nn.dropout(fc, self.dict_model["keep"], name="{}/dropout".format(name))
            
        if activation == "relu":
            fc = tf.nn.relu(fc, name="{}/relu".format(name))
        elif activation == "sigmoid":
            fc = tf.nn.sigmoid(fc, name="{}/sigmoid".format(name))
        
        return fc

    def avgpool(self, tensor, k=2, d=1):
        if len(tensor.shape) == 5:
            return tf.nn.avg_pool3d(tensor, ksize=[1,  d,k, k, 1], 
                                    strides=[1, d , k, k, 1], 
                                    data_format= "NDHWC" , padding='VALID')
        
        return tf.nn.avg_pool(tensor, ksize=[1, k, k, d], strides=[1, k, k, d], padding='VALID' )
                
    def maxpool(self, tensor, k=2, d=1, name = "pool"):
        if len(tensor.shape) == 5:
            return tf.nn.max_pool3d(tensor, ksize=[1,  d,k, k, 1], 
                                    strides=[1, d , k, k, 1], 
                                    data_format= "NDHWC" , padding='VALID', name = name)
        
        return tf.nn.max_pool(tensor, ksize=[1, k, k, d], strides=[1, k, k, d], padding='VALID', name = name )
                                                    
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
        
    def conv2d(self, x, out_channels, kernel_size,  name = "conv",  batch_norm = False, padding = "SAME"):
        depth = x.shape.as_list()[-1]
        w, b = self.get_weights(depth, out_channels,kernel_size, name)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], data_format= "NHWC", padding = padding)
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        if batch_norm:
            x = self.batch_norm(x, "{}/batch_norm" % name)
            
        return x 
    
    def repeat(self, num_layers_repeateds, layer_fn, input, out_channels, kernel_size, name = "conv",  
               batch_norm = False, padding = "SAME"):
        net = input
        for i in range(1,num_layers_repeateds+1):
            net = layer_fn(net, out_channels, kernel_size, "{}/{}_{}".format( name,name,i), batch_norm, padding)
        return net
    
    def batch_norm(self, tensor, name = None):
         return  tf.layers.batch_normalization(tensor, fused=True, data_format='NCHW', name = name, 
                                                 training = self.param.is_training)

    def fc_cosnorm(self,x, w, biases, bias=0.00001, name = "cosNorm"):
        x = tf.add(x, bias)
        w = tf.add(w, bias)

        y = tf.matmul(x, w) + biases

        x = tf.reduce_sum(tf.square(x),1, keepdims=True)
        x = tf.sqrt(x)

        w = tf.reduce_sum(tf.square(w),0, keepdims=True)
        w = tf.sqrt(w)


        return tf.divide(y ,(x * w), name = name)

    def regularize(self, loss, type = 1, scale = 0.005, scope = None):
        if type == 1:
            regularizer = tf.contrib.layers.l1_regularizer( scale=scale, scope=scope)
        else:
            regularizer = tf.contrib.layers.l2_regularizer( scale=scale, scope=scope)
                
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
        regularized_loss = loss + regularization_penalty
        return regularized_loss     

    def _new_fc_model(self, input_layer):
            hidden = self.hparams.hidden_layers
            fc = input_layer
            id = 1
            
            for num_neurons in hidden: 
                fc = self.fully_connected(input = fc, hidden_units = num_neurons,   batch_norm = self.hparams.batch_norm, 
                                          name = "fc{}".format(id) )
                id += 1

            return self.logits_layer(fc)
   
    def add_fully_connected(self, flatten):
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            out_list = flatten.shape.as_list()
            BOTTLENECK_TENSOR_SIZE = np.prod(out_list[1:]) 
            with tf.name_scope('bottleneck'):
                bottleneck_tensor = flatten
                bottleneck_input = tf.placeholder(tf.float32,
                shape=[None, BOTTLENECK_TENSOR_SIZE],
                name='InputPlaceholder')

            with tf.name_scope('fully_connected'):
                logits = self._new_fc_model(bottleneck_input) 
                self.dict_model["bottleneck_tensor"] = bottleneck_tensor
                self.dict_model["bottleneck_input"] = bottleneck_input

        else:
            with tf.name_scope('fully_connected'):
                 logits = self._new_fc_model(flatten)
        
        return logits
    
    def logits_layer(self, fc_layer):
        out_shape = fc_layer.shape.as_list()
        out_size = np.prod(out_shape[1:])
        w, b = self.get_weights(out_size, self.hparams.num_classes, name = "logits/weight")
        logits = tf.add(tf.matmul(fc_layer, w), b, name="logits")
        return logits
    
    
    def _transfer_learning(self, inputs, only_cnn = False):
        vars = set([ var.name for var in tf.all_variables()])
        file_name = self.info["file_name"]
        main_directory = "./models/"

        model_path = os.path.join(main_directory,file_name)

        if not os.path.exists(model_path):
            utils.maybe_download_and_extract(self.info["url"], main_directory, file_name, file_name)

        cnn = self._get_cnn(inputs)

        if not self.hparams.fine_tunning:
            print("Stopping gradient")
            cnn = tf.stop_gradient(cnn) #Just use it in case of transfer learning without fine tunning


        if not only_cnn:
            
            flatten = tf.layers.flatten(cnn, name="flatten")
            cnn = self._get_fully_connected(flatten)


        new_vars = list(set([ var.name for var in tf.all_variables()]) - vars)

        self.vars_to_restore = new_vars




          # Check the values of the variables

      #   print(graph.get_operations())
        return cnn
    
        
    def generate_optimization_parameters(self,logits):
        with tf.device("cpu:0"):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.hparams.initial_learning_rate, global_step, 
                                               self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)  
        with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self._labels))
                if self.hparams.regularizer_type:
                    loss = self.regularize(loss, self.hparams.regularizer_type, self.hparams.regularizer_scale)
                tf.summary.scalar("loss", loss)

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.name_scope('train_accuracy'):
            acc = tf.equal(tf.argmax(logits, 1), tf.argmax(self._labels, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            tf.summary.scalar("accuracy", acc)


        predictions = {
                       "classes": tf.argmax(logits, 1),
                       "probs" :  tf.nn.softmax(logits), 
                       "labels": tf.argmax(self._labels, 1)
                       }
        
        
        self.dict_model["loss"] = loss
        self.dict_model["optimizer"] = optimizer
        self.dict_model["accuracy"] = acc
        self.dict_model["predictions"] = predictions
        self.dict_model["labels"] = self._labels
        self.dict_model["global_step"] = global_step
    
    def _generate_checkpoint_path(self):
        ckpt_path = os.path.join(self.hparams.checkpoints_path, self.dataset.get_name(),"{}_{}".format(self.get_name(),
                                                                                                     self.hparams.cut_layer)
                                )
        return ckpt_path
                                 
    def create_monitored_session(self, model,iter_per_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


#         sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
#                                             save_checkpoint_secs=120,
#                                             log_step_count_steps=iter_per_epoch,
#                                             save_summaries_steps=iter_per_epoch,
#                                             config=config) 
        sess = tf.Session(config=config)
        all_vars = tf.all_variables()
        saver = tf.train.Saver( var_list = all_vars, max_to_keep=1)
        
        ckpt_path = self._generate_checkpoint_path()
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            
        if len(os.listdir(ckpt_path))  > 0:
            
            saver.restore(sess, tf.train.latest_checkpoint(self.hparams.checkpoints_path))
             
            print("The entire model was restored.")
        else:
            
            to_restore =  [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.info["model_name"]) if var.name in self.vars_to_restore ]
            
            res = tf.train.Saver(to_restore)
            
          
            
            try:
                
                model_path = "./models/{}".format(self.info["file_name"])
                print("Restoring variables from: {}".format(model_path))
                res.restore(sess, model_path)
                restored = [var.name for var in to_restore]


                unrestored = [var for var in all_vars if var.name not in restored]

                print("Initializing only unrestored variables")
                sess.run(tf.initialize_variables(unrestored))
  
            except:
                print("Error in the restore process. Initializing from scratch instead!")
                sess.run(tf.initialize_variables(all_vars))
           
            
                  
        
        return sess, saver
    
    def _run_optmizer_training(self,saver, model, input_data_placeholder, steps_per_epoch):
        ckpt_path = os.path.join(self._generate_checkpoint_path(),"model")
        msg = "--> Global step: {:>5} - Mean acc: {:.2f}% - Batch_loss: {:.4f} - ({:.2f}, {:.2f}) (steps,images)/sec"
        best_accuracy = 0
        total_images = 0
        
        if  self.dataset.istfrecord():
            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            try:
                    
                feed_dict={model["keep"]:self.hparams.keep}
                for epoch in range(self.hparams.num_epochs):
                    start_time = time()

                    print("\n*************************************************************")
                    print("Epoch {}/{}".format(epoch+1, self.hparams.num_epochs))

                    sum_acc= 0
                    for s in range(steps_per_epoch):
                        _, batch_loss, batch_acc, step = self.sess.run(
                        [model["optimizer"], model["loss"], model["accuracy"], model["global_step"]], feed_dict=feed_dict)
                        sum_acc += batch_acc * self.hparams.batch_size

                    duration = time() - start_time
                    mean_acc = sum_acc / (steps_per_epoch * self.hparams.batch_size)
                    print(msg.format(step,  mean_acc*100, batch_loss, (steps_per_epoch / duration), 
                                     (steps_per_epoch*self.hparams.batch_size / duration) 
                                    ))

                            
                    
            except tf.errors.OutOfRangeError:
                print('Done training... An error was ocurred during the training!!!')
            finally:
                coord.request_stop()

            coord.join(threads)
            print('Done training...')
 
        
        else:
   
            train_data, train_labels = self.dataset.get_train()

            val_data, val_labels = self.dataset.get_validation()
            for epoch in range(self.hparams.num_epochs):
                start_time = time()

                print("\n*************************************************************")
                print("Epoch {}/{}".format(epoch+1, self.hparams.num_epochs))

                sum_acc= 0
                for s in range(steps_per_epoch):

                    batch_data, batch_labels = self.dataset.next_batch()
                    feed_dict={input_data_placeholder: batch_data, model["labels"]: batch_labels, model["keep"]:self.hparams.keep}

                    _, batch_loss, batch_acc, step = self.sess.run(
                    [model["optimizer"], model["loss"], model["accuracy"], model["global_step"],],
                    feed_dict=feed_dict)
                    sum_acc += batch_acc * self.hparams.batch_size

                duration = time() - start_time
                mean_acc = sum_acc / (steps_per_epoch * self.hparams.batch_size)
                print(msg.format(step,  mean_acc*100, batch_loss, (steps_per_epoch / duration), 
                                 (steps_per_epoch*self.hparams.batch_size / duration) 
                                ))

               
                acc = self._evaluate( val_data, val_labels )

                if acc > best_accuracy:

                    saver.save(self.sess,ckpt_path, global_step=step)
                    if best_accuracy  > 0:
                        print "The model was saved. The current evaluation accuracy ({}%) is greater than the last one saved({}%).".format(acc*100,best_accuracy*100)
                    else:
                        print "The model was saved."
                    
                    best_accuracy= acc
    
    def train(self):
        if self.hparams is None:
            raise "Hyperparameters object was not supplied."
        if not self.built:
            self.build()
        train_data, train_labels = self.dataset.get_train()
#         train_data, train_labels = utils.data_augmentation(train_data, train_labels)

        model = self.dict_model


        steps_per_epoch = int(math.ceil(len(train_data) /  self.hparams.batch_size))
        
       
        if self.sess is None:
            self.sess, saver = self.create_monitored_session(model, steps_per_epoch)

        
        
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            indices = list( range(len(train_data)) )
            

            self.dataset.get_or_generate_bottleneck(sess = self.sess, model = model,
                                                             file_name = "bottleneck_{}_{}_{}_train".format( self.dataset.get_name(),
                                                                                             self.info["model_name"],
                                                                                             self.hparams.cut_layer),
                                                    type = "train"
                                              )
            
            self.dataset.get_or_generate_bottleneck(self.sess, model, 
                                                               "bottleneck_{}_{}_{}_test".format( self.dataset.get_name(),
                                                                                              self.info["model_name"],
                                                                                              self.hparams.cut_layer), 
                                               type = "test")
            
            self.dataset.get_or_generate_bottleneck(self.sess, model, 
                                                               "bottleneck_{}_{}_{}_validation".format( self.dataset.get_name(),
                                                                                              self.info["model_name"],
                                                                                              self.hparams.cut_layer), 
                                               type = "validation")


            input_data_placeholder = model["bottleneck_input"]

        else:
            input_data_placeholder = model["images"]
            
            
        self._run_optmizer_training(saver, model, input_data_placeholder, steps_per_epoch)
        self.plot_data()
            
            
            
        
        

        
    def _evaluate(self, data, labels, batch_size = 100):
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            input_data_placeholder = self.dict_model["bottleneck_input"]
        else:
            input_data_placeholder = self.dict_model["images"]

        if self.sess is None:
            self.sess = self.create_monitored_session(self.dict_model,steps_per_epoch)

        size = len(data)//batch_size
        indices = list(range(len(data)))
        global_acc = 0;

        for i in range(size+1):

            begin = i*batch_size
            end = (i+1)*batch_size
            end = len(data) if end >= len(data) else end
             
            next_bach_indices = indices[begin:end]
            if len(next_bach_indices) == 0:
                break;
                
            batch_data = data[next_bach_indices]
            batch_labels = labels[next_bach_indices]
            

            acc = self.sess.run(self.dict_model["accuracy"],
                feed_dict={input_data_placeholder: batch_data, self.dict_model["labels"]: batch_labels, self.dict_model["keep"]:1.0})

            global_acc += (acc * len(next_bach_indices))
            
      
        mes = "--> Evaluation accuracy: {:.2f}%"
        global_acc /= len(data)
        print(mes.format(global_acc * 100))
       
        return global_acc


    def test_prediction(self, batch_size = 128):
        data, labels = self.dataset.get_test()
        if not self.built:
            self.build()
            
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            input_data_placeholder = self.dict_model["bottleneck_input"]
        else:
            input_data_placeholder = self.dict_model["images"]

        

        model = self.dict_model

        

        if self.sess is None:
            self.sess,_ = self.create_monitored_session(self.dict_model,100)
        predictions = {
                       "classes":[],
                       "probs":[],
                       "labels":[]
                      }

        size = len(data)//batch_size
        indices = list(range(len(data)))

        for i in range(size+1):

            begin = i*batch_size
            end = (i+1)*batch_size
            end = len(data) if end >= len(data) else end

            next_bach_indices = indices[begin:end]
            batch_xs = data[next_bach_indices]
            batch_ys = labels[next_bach_indices]

            pred = self.sess.run(self.dict_model["predictions"],
                feed_dict={input_data_placeholder: batch_xs, self.dict_model["labels"]: batch_ys, self.dict_model["keep"]:1.0})

            predictions["classes"].extend(pred["classes"])
            predictions["probs"].extend(pred["probs"])
            predictions["labels"].extend(pred["labels"])


        correct = list (map(lambda x,y: 1 if x==y else 0, predictions["labels"] , predictions["classes"]))
        acc = np.mean(correct ) *100

        mes = "--> Prediction accuracy: {:.2f}% ({}/{})"
        print(mes.format( acc, sum(correct), len(data)))

        return predictions
