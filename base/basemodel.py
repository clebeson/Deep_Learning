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
import functions as  bf
import layers
from base.baselayer import BaseLayer
from texttable import *
from hurry.filesize import size, si


class BaseModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_cnn(self, inputs, batch_norm = None ):  pass
    
    
    def _get_fully_connected(self, batch_norm = False): pass
    
    
    
    
        
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
                      "bottleneck_input":None,
                      "layers":{}
                  }
        self.vars_to_restore = []
        self._ckpt_dir = "./ckpts"
        self._layers = []
 
        
    def add(self, layer, *args, **kwargs):
        
        if not issubclass(layer, BaseLayer):
            raise Exception("The layer must be a ""BaseModel"" subclass rather than {}".format(type(layer)))
        
        keys = kwargs.keys()
        input = self.dict_model["images"] if len(self._layers) == 0 else self._layers[-1]
        
        if "name" in keys:
            name = kwargs["name"]
            del kwargs["name"]
        else:
            name = layer.__name__.lower()
        
        net = layer(input, *args, name= name, **kwargs)  
        self._layers.append(net)
        self.dict_model["layers"].update({net.name: net})
        
        
    def add_repeated(self, num_repetition, layer, *args, **kwargs):
        
        if not issubclass(layer,BaseLayer):
            raise Exception("The layer must be a ""BaseModel"" type rather than {}".format(type(layer)))
        
        keys = kwargs.keys()
        net = self._input_data if len(self._layers) == 0 else self._layers[-1]
        if "name" in keys:
            name = kwargs["name"]
            del kwargs["name"]
        else:
            name = layer.__name__
        
      
        for i in range(1,num_repetition+1):         
            net = layer(net, *args, name= "{}/{}_{}".format( name,name,i), **kwargs) 
            self._layers.append(net)
            self.dict_model["layers"].update({net.name: net})
        
            
    def activation_maps(self,images=None):
        if self.hparams is None:
            raise Exception("Hyperparameters object was not supplied.")
        if not self.built:
            self.build()

       
        model = self.dict_model
        steps_per_epoch = 1
        
       
        if self.sess is None:
            self.sess, _ = self.create_monitored_session(model, steps_per_epoch)
       
        images = self.dataset.get_sample() if not images or len(images) == 0 else images
        
        for i, image in enumerate(images):
            image = np.expand_dims(image, axis=0) if len(list(image.shape)) == 3 else image
            utils.grad_CAM_plus(self.sess, image, model, i, "image_{}.jpg".format(i))

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
                    self._input_data, self._labels = self.dataset.get_tfrecord_data()
                    self.hparams.bottleneck = False
    #                     input_images =  self.input_data
                else:
                    self._input_data = tf.placeholder(tf.float32, shape=[None,self.hparams.height, 
                                                              self.hparams.width, 
                                                              self.hparams.channels], 
                                                              name='images')

                    self._labels = tf.placeholder(tf.float32, shape=[None, self.hparams.num_classes], name='labels')
                    self.dict_model["images"] = self._input_data 
            self.dict_model["keep"] = tf.placeholder(tf.float32, name='dropout_keep')

            


            
                
            with tf.name_scope(self.info["model_name"]):
               # Create a VGG16 model and reuse its weights.
                self._transfer_learning(self._input_data, only_cnn = new_fc) if  is_tf else self._from_scratch(
                    self._input_data, only_cnn = new_fc)
                       
            if  new_fc:
#                 with tf.name_scope("flatten"):
#                     self.add(layers.Flatten, name="flatten")

                self.add_fully_connected()
      
            self.generate_optimization_parameters()
            self.built = True
#             g = tf.get_default_graph()
#             for i in g.get_operations():
#                 print(i.values())
           
            
        
    def get_name(self):
        return self.info["model_name"]

    
                    

    def _new_fc_model(self):
            hidden = self.hparams.hidden_layers
            id = 1
            keep =   None  if(not self.hparams.keep or self.hparams.keep  == 1.0) else self.dict_model["keep"]
            
            for num_neurons in hidden: 
                self.add(layers.FullyConnected, hidden_units = num_neurons, keep = keep,  
                                        norm = self.hparams.normalization, 
                                        name = "fc{}".format(id) )
                id += 1

            self.add(layers.Logits, num_classes = self.hparams.num_classes)
   
    def add_fully_connected(self):
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            out_list = self._layers[-1].shape.as_list()
            BOTTLENECK_TENSOR_SIZE = np.prod(out_list[1:]) 
            with tf.name_scope('bottleneck'):
                bottleneck_tensor = self._layers[-1]
                bottleneck_input = tf.placeholder(tf.float32,
                shape=[None, BOTTLENECK_TENSOR_SIZE],
                name='InputPlaceholder')

            with tf.name_scope('fully_connected'):
                self._new_fc_model() 
                self.dict_model["bottleneck_tensor"] = bottleneck_tensor
                self.dict_model["bottleneck_input"] = bottleneck_input

        else:
            with tf.name_scope('vgg_16'):
                 self._new_fc_model()
        
       

    
    def _transfer_learning(self, inputs, only_cnn = False):
        vars = set([ var.name for var in tf.all_variables()])
        file_name = self.info["file_name"]
        main_directory = self._ckpt_dir

        model_path = os.path.join(main_directory,file_name)

        if not os.path.exists(model_path):
            utils.maybe_download_and_extract(self.info["url"], main_directory, file_name, file_name)

        self._get_cnn(inputs,  self.hparams.normalization)

        if not self.hparams.fine_tunning:
            print("Stopping gradient")
            self._layers[-1].stop_gradient() #Just use it in case of transfer learning without fine tunning


        if not only_cnn:
            
#             self.add(layers.Flatten, name="flatten")
            self._get_fully_connected(self.hparams.normalization)


        new_vars = list(set([ var.name for var in tf.all_variables()]) - vars)

        self.vars_to_restore = new_vars

      #   print(graph.get_operations())
        
    
        
        
    def summary(self):
        rows = [["Name", "type", "input", "Feature Maps", "kernel", "stride", "Parameters"]]
        total = 0.0
        fc = 0.0
        for l in self._layers:
            row = l.summary()
            total += row[-1]
            
            fc  += row[-1] if l.type == "fc" or l.type == "fc_conv" else 0.0
            row[-1] = size(row[-1], system=si) if row[-1] >= 1000 else row[-1]
            rows.append(row)
        
        table = Texttable()
        table.set_cols_align(["c", "c", "c", "c", "c", "c", "c"])
        table.set_cols_valign(["m", "m", "m", "m", "m", "m", "m"])
        table.add_rows(rows)
        print table.draw()
        percent_fc = fc/total
        print "Total Parameters: {}   Percents: (CNN-{:.2f}%)  (FC-{:.2f}%)".format(size(total, system=si), (1.0-percent_fc)*100, percent_fc*100)
    
    def generate_optimization_parameters(self):
        logits = self._layers[-1].output
        with tf.device("cpu:0"):
            
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.hparams.initial_learning_rate, global_step,
                                                       self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)
        with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self._labels))
                if self.hparams.regularizer_type:
                    loss = regularize(loss, self.hparams.regularizer_type, self.hparams.regularizer_scale)
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
    
    
    def info_visualization(self, name_layer = None):

        if self.hparams is None:
            raise Exception("Hyperparameters object was not supplied.")
        if not self.built:
            self.build()

        layer = None
        if name_layer is None: layer = self._layers[1]
        else:
            
            for l in self._layers:
                if l.name == name_layer:
                    layer = l
                    break
        if layer is None:
            print("A valid name of layer is requirede")
            return
        
        model = self.dict_model

#         for op in tf.get_default_graph().get_operations():
#             print str(op.name) 
            
#         return
        steps_per_epoch = 1
        
       
        if self.sess is None:
            self.sess, _ = self.create_monitored_session(model, steps_per_epoch)
        
        image = self.dict_model["images"]

        name = layer.output.name
#         name = "/".join(name.split("/")[:-1])+":0"
        tensor = tf.get_default_graph().get_tensor_by_name(name)
        label = np.zeros((1,10))*1.0
        images = []
        tensor_shape = tensor.shape.as_list()
        for feature in range(tensor_shape[-1]):
            loss = tf.reduce_mean(tensor[:,:,:,feature]) if len( tensor_shape) == 4 else tensor[0,feature]
            gradient = tf.gradients(loss, self.dict_model["images"])
            image_shape = image.shape.as_list()
            image_shape[0] = 1
#             print loss, gradient, tensor, tensor_shape, tensor[0,feature]
            print("Feature: {}".format(feature))            
            feature_image = np.random.uniform(size=image_shape) + 128.0
            for i in range(30):
                feed_dict = {self.dict_model["images"]: feature_image, self.dict_model["labels"]: label}
                grad, loss_value = self.sess.run([gradient, loss],feed_dict=feed_dict)
                grad = np.array(grad).squeeze()
                step_size = 1.0 / (grad.std() + 1e-8)
                feature_image += step_size * grad
                feature_image = np.clip(feature_image, 0.0, 255.0)
                
            images.append(feature_image.squeeze())
            
        return images
            
        

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
            
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
             
            print("The entire model was restored.")
        else:
            
            to_restore =  [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.info["model_name"]) if var.name in self.vars_to_restore ]
            
            res = tf.train.Saver(to_restore)
            
          
            
            try:
                
                model_path = "{}/{}".format(self._ckpt_dir, self.info["file_name"])
                print("Restoring variables from: {}".format(model_path))
                res.restore(sess, model_path)
                restored = [var.name for var in to_restore]


                unrestored = [var for var in all_vars if var.name not in restored]

                print("Initializing only unrestored variables")
                sess.run(tf.initialize_variables(unrestored))
  
            except Exception as e: 
                print("Error in the restore process. Initializing from scratch instead!")
                sess.run(tf.initialize_variables(all_vars))
                
                print("\n***** Error message *******")
                print("--> {}".format( e.message))
                print("*****************************\n")
           
            
                  
        
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
                        print "The model was saved. The current evaluation accuracy ({:.2f}%) is greater than the last one saved({:.2f}%).".format(acc*100,best_accuracy*100)
                    else:
                        print "The model was saved."
                    
                    best_accuracy= acc
    
    def train(self):
        if self.hparams is None:
            raise Exception("Hyperparameters object was not supplied.")
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
#         self.plot_data()
            

        
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
