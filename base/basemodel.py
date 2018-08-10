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
import layers
from base.baselayer import BaseLayer
from texttable import *
from hurry.filesize import size, si
from scipy.ndimage.filters import gaussian_filter as gauss
import sklearn.metrics as sklm

class BaseModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _create_cnn(self, inputs, batch_norm = None ):  pass
    
    @abstractmethod
    def _create_fully_connected(self, batch_norm = False): pass
    
    
    
    
        
    def __init__(self,info, dataset, cut_layer = None ):
        self.cut_layer = cut_layer
        self.hparams = dataset.hparams
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
                      "keep": None,
                      "bottleneck_tensor":None,
                      "bottleneck_input":None,
                      "layers":{}
                  }
        self.vars_to_restore = []
        self._ckpt_dir = "./models/ckpts"
        self._layers = []
        self._istraining = False
 
    def get_layers(self, id = None):
        if id is None:
            return self._layers
        if type(id) == str:
            for l in self._layers:
                if l.name == id:
                    return l
        return self._layers[id]
    
    def add(self, layer, *args, **kwargs):
        
        if not issubclass(layer, BaseLayer):
            raise Exception("The layer must be a ""BaseModel"" subclass rather than {}".format(type(layer)))
            
        keys = kwargs.keys()
        if not self._layers:
            input = self.dict_model["images"]
        elif "input" in keys:
            input = kwargs["input"]
            kwargs.pop("input")
        else:
            input =  self._layers[-1]
        
        
        if args:
             net = layer(input,*args, **kwargs)  
        else:
            if "name" not in keys:
                kwargs["name"] = layer.__name__.lower()
                
            kwargs["input"] = input  
            net = layer(**kwargs)  
       
        net._my_input_data_placeholder = self.dict_model["images"]       
        self._layers.append(net)
        self.dict_model["layers"].update({net.name: net})
        
        
    def add_repeated(self, num_repetition, layer, *args, **kwargs):

        if "name" in kwargs.keys():
            name = kwargs["name"]
        else:
            name = layer.__name__

        for i in range(1,num_repetition+1):
            
            kwargs["name"] = "{}/{}_{}".format( name,name,i)
            self.add(layer, *args, **kwargs)

        
            
  
    def _build_only_cnn(self, input_images, istraining, is_tf = True):
            self.close()
            print("Buiding cnn from ""{}"" model.".format(self.info["model_name"]) )
            self.dict_model["images"] = input_images
            self.dict_model["istraining"] = istraining     
            with tf.name_scope(self.info["model_name"]):
                cnn  = self._transfer_learning(input_images, only_cnn = True) if  is_tf else self._from_scratch(input_images, 
                                                                                                                only_cnn = True) 
            for l in self._layers:
                l._model_name = self.info["model_name"]
                l._istraining_placeholder = istraining
                
            self.built = True
            self.vars_to_restore = np.reshape([ [l.weights.name, l.biases.name] for l in self._layers if l.weights is not None], (-1,1))



    def close(self):
        for k in list(self.dict_model):
            if k == "layers":
                 for l in list(self.dict_model[k]):
                        del self.dict_model[k][l]   
            del self.dict_model[k]
        for var in self.vars_to_restore:
            del var
       
            
        self.dict_model = {
                      "global_step": None,
                      "images": None,
                      "labels": None,    
                      "loss" : None,
                      "optimizer": None,
                      "accuracy": None,
                      "keep": None,
                      "bottleneck_tensor":None,
                      "bottleneck_input":None,
                      "layers":{}
                  }
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        self._layers = []

        self.built = False
        
        
        
   
        
    def build(self, is_tf = True, new_fc = True, istraining = False):
            self.close()
            tf.reset_default_graph()

            self._istraining = istraining
   
            if  not new_fc:
                self.hparams.cut_layer = self.info["last_layer"]
                self.hparams.fine_tunning = True
                
            with tf.device("cpu:0"):
                if self.dataset.istfrecord():
                    input_data, labels = self.dataset.get_tfrecord_data()
                    self.hparams.bottleneck = False
    #                     input_images =  self.input_data
                else:
                    h, w = self.hparams.height, self.hparams.width
                    if self.hparams.data_augmentation and not self.hparams.auto_crop == (0,0): 
                        
                        h = self.hparams.height - self.hparams.auto_crop[0]
                        w = self.hparams.width - self.hparams.auto_crop[1]
                    input_data = tf.placeholder(tf.float32, shape=[None,h, w, self.hparams.channels], name='images')
                    
                    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
                self.dict_model["images"] = input_data 
                self.dict_model["labels"] = labels
            self.dict_model["keep"] = tf.placeholder(tf.float32, name='dropout_keep')
            self.dict_model["istraining"] = tf.placeholder(tf.bool, name='istraining')
             
#             print("***********************************") 
#             print(self.dict_model)
#             g = tf.get_default_graph()
#             for i in g.get_operations():
#                 print(i.values())
                
            with tf.name_scope(self.info["model_name"]):
               # Create a VGG16 model and reuse its weights.
                self._transfer_learning(input_data, only_cnn = new_fc) if  is_tf else self._from_scratch(
                    input_data, only_cnn = new_fc)
                       
            if  new_fc:
                with tf.name_scope("flatten"):
                    self.add(layers.Flatten, name="flatten")

                self.add_fully_connected()
            
            if istraining:
                self.generate_optimization_parameters()
                
        
            if self.sess is None:
                self.sess, self._saver = self.create_monitored_session(self.dict_model)
       
            for l in self._layers:
                l._model_input_data_placeholder = self.dict_model["images"]       
                l._labels_placeholder = self.dict_model["labels"]
                l._keep_placeholder= self.dict_model["keep"]
                l._istraining_placeholder = self.dict_model["istraining"]
                l._sess = self.sess
                l._logits = self._layers[-1]
            
            
           
            self.built = True
            

            

           
            
        
    def get_name(self):
        return self.info["model_name"]

    
                    

    def _new_fc_model(self, input = None):
            hidden = self.hparams.hidden_layers
            id = 1
            keep =   None  if(not self.hparams.keep or self.hparams.keep  == 1.0) else self.dict_model["keep"]

            for i, num_neurons in enumerate(hidden): 
                if i == 0 and input is not None:
                    
                    self.add(layers.FullyConnected, hidden_units = num_neurons, istraining = self.dict_model["istraining"], 
                             input = input, keep = keep, norm = self.hparams.normalization,
                             name = "fc{}".format(id))
                else:
                    self.add(layers.FullyConnected, hidden_units = num_neurons, istraining = self.dict_model["istraining"],
                             keep = keep, norm = self.hparams.normalization, name = "fc{}".format(id))
                id += 1

            self.add(layers.Logits, num_classes = self.hparams.num_classes)
   

    def add_fully_connected(self):
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            out_list = self._layers[-1].shape.as_list()
            with tf.name_scope('bottleneck'):
                bottleneck_tensor = self._layers[-1].output
                bottleneck_input = tf.placeholder(tf.float32,
                shape=[None]+out_list[1:],
                name='InputPlaceholder')
                self.dict_model["bottleneck_tensor"] = bottleneck_tensor
                self.dict_model["bottleneck_input"] = bottleneck_input

            with tf.name_scope('fully_connected'):
                self._new_fc_model(input = bottleneck_input) 


        else:
            with tf.name_scope('vgg_16'):
                 self._new_fc_model()
        
      

    
    def _transfer_learning(self, inputs, only_cnn = False):

        file_name = self.info["file_name"]
        main_directory = self._ckpt_dir

        model_path = os.path.join(main_directory,file_name)

        if not os.path.exists(model_path):
            utils.maybe_download_and_extract(self.info["url"], main_directory, file_name, file_name)

        self._create_cnn(inputs,  self.hparams.normalization)

        if not self.hparams.fine_tunning:
            print("Stopping gradient")
            self._layers[-1].stop_gradient() #Just use it in case of transfer learning without fine tunning

        
        if not only_cnn:
            
#             self.add(layers.Flatten, name="flatten")
            self._create_fully_connected(self.hparams.normalization)
    
        self.vars_to_restore = np.reshape([ [l.weights.name, l.biases.name] for l in self._layers if l.weights is not None], (-1,1))
            
          
      #   print(graph.get_operations())
        
    
        
        
    def summary(self):
        rows = [["Name", "type", "input", "Feature Maps", "kernel", "stride", "Parameters"]]

        total = 0.0
        fc = 0.0
        for l in self._layers:
            
            row = l.summary()
        
            if row[0] is not None:
                row = ["({}){}".format(*row[:2])]+row[2:]
            else:
                row = row[1:]  
            total += row[-1]
            
            fc  += row[-1] if l.type == "fc" or l.type == "fc_conv" else 0.0
            row[-1] = size(row[-1], system=si) if row[-1] >= 1000 else row[-1]
            rows.append(row)
        
        table = Texttable()
        table.set_cols_align(["c", "c", "c", "c", "c", "c", "c"])
        table.set_cols_valign(["m", "m", "m", "m", "m", "m", "m"])
        table.add_rows(rows)
        print(table.draw())
        percent_fc = fc/total
        print("Total Parameters: {}   Percents: (CNN-{:.2f}%)  (FC-{:.2f}%)".format(size(total, system=si), (1.0-percent_fc)*100, percent_fc*100) )
    
    def generate_optimization_parameters(self):
        logits = self._layers[-1].output
        with tf.device("cpu:0"):
            
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.hparams.initial_learning_rate, global_step,
                                                       self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)
        with tf.name_scope('train_accuracy'):
            pred, lab = tf.argmax(logits, 1), self.dict_model["labels"]
#             _,acc = tf.metrics.accuracy(predictions = pred, labels = lab)
            acc = tf.equal(tf.argmax(logits, 1), tf.cast(lab,  tf.int64))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            _,precision = tf.metrics.precision(predictions = pred, labels = lab)
            _,recall = tf.metrics.recall(predictions = pred, labels = lab)
            f1_score = (2 * (precision * recall)) / (precision + recall)

#             tf.summary.scalar("accuracy",recall)
        class_weights = tf.multiply(0.0, tf.cast(tf.equal(lab, 1), tf.float32)) + 1
       
        with tf.name_scope('loss'):
#                 loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=self.dict_model["labels"], weights = class_weights))
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                                 labels=tf.one_hot(self.dict_model["labels"],
                                                                                                   self.hparams.num_classes)))

                if self.hparams.regularizer_type:
                    loss = self.regularize(loss, self.hparams.regularizer_type, self.hparams.regularizer_scale)
                #                 loss += loss/(precision*recall+1e-5)
                tf.summary.scalar("loss", loss)

        with tf.name_scope('sgd'):
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        
        self.dict_model["loss"] = loss
        self.dict_model["optimizer"] = optimizer
        self.dict_model["metrics"] = {"accuracy":acc,"precision":precision,"":recall,"f1_score":f1_score}
        self.dict_model["global_step"] = global_step
        self.dict_model["logits"] = logits
        
    
    def regularize(self, loss, type = 1, scale = 0.005, scope = None):
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

    
            
        
    
    def _generate_checkpoint_path(self):
        ckpt_path = os.path.join(self.hparams.checkpoints_path, self.dataset.get_name(),"{}_{}".format(self.get_name(),
                                                                                                     self.hparams.cut_layer)
                                )
        return ckpt_path
                                 
    def create_monitored_session(self, model,iter_per_epoch = 1):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


#         sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
#                                             save_checkpoint_secs=120,
#                                             log_step_count_steps=iter_per_epoch,
#                                             save_summaries_steps=iter_per_epoch,
#                                             config=config) 
        sess = tf.Session(config=config)
        all_vars = tf.global_variables()
        all_vars.extend(tf.local_variables())
        saver = tf.train.Saver(max_to_keep=1)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)

        ckpt_path = self._generate_checkpoint_path()
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            
        if len(os.listdir(ckpt_path))  > 0:
            
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
             
            print("The entire model was restored.")
        else:
            
            to_restore =  [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.info["model_name"]) if var.name in self.vars_to_restore ]
    
            
            try:
                if to_restore:
                    res = tf.train.Saver(to_restore)
                    model_path = "{}/{}".format(self._ckpt_dir, self.info["file_name"])
                    print("Restoring variables from: {}".format(model_path))
                    res.restore(sess, model_path)


    
                else:
                 print("Initializing variables")
#                  sess.run(tf.initialize_variables(all_vars))
                   
            except Exception as e: 
                print("Error in the restore process. Initializing from scratch instead!")
                
                
                print("\n***** Error message *******")
                print("--> {}".format( e.message))
                print("*****************************\n")
           
            
                  
        
        return sess, saver
    
    def _run_optmizer_training(self, model, input_data_placeholder, steps_per_epoch):
        ckpt_path = os.path.join(self._generate_checkpoint_path(),"model")
        msg = "--> Global step: {:>5} - Mean acc: {:.2f}% - Batch_loss: {:.4f} - ({:.2f}, {:.2f}) (steps,images)/sec"
        best_accuracy = 0
        total_images = 0
        
        if  self.dataset.istfrecord():
            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            try:
                    
                feed_dict={model["keep"]:self.hparams.keep, self.dict_model["istraining"]:True}
                step = 0
                for epoch in range(self.hparams.num_epochs):
                    start_time = time()

                    print("\n*************************************************************")
                    print("Epoch {}/{}".format(epoch+1, self.hparams.num_epochs))

                    sum_acc= 0
                    for s in range(steps_per_epoch):
                        _, batch_loss, batch_acc, step = self.sess.run(
                        [model["optimizer"], model["loss"], model["metrics"]["accuracy"], model["global_step"]], feed_dict=feed_dict)
                        sum_acc += batch_acc * self.hparams.batch_size
                    
                    duration = time() - start_time
                    mean_acc = sum_acc / (steps_per_epoch * self.hparams.batch_size)
                    print(msg.format(step,  mean_acc*100, batch_loss, (steps_per_epoch / duration), 
                                     (steps_per_epoch*self.hparams.batch_size / duration) 
                                    ))
                    if epoch %5 == 0:
                        print("The model was saved.")
                        self._saver.save(self.sess,ckpt_path, global_step=step)
                        

                            
            except tf.errors.OutOfRangeError:
                print('Done training... An error was ocurred during the training!!!')
            finally:
#                 self._saver.save(self.sess,ckpt_path, global_step=step)        
                coord.request_stop()
            
            coord.join(threads)
            print('Done training...')
 
        
        else:
            
            val_data, val_labels = self.dataset.get_test()
           
            for epoch in range(self.hparams.num_epochs):
                start_time = time()

                print("\n*************************************************************")
                print("Epoch {}/{}".format(epoch+1, self.hparams.num_epochs))

                sum_acc= 0
                for s in range(steps_per_epoch):

                    batch_data, batch_labels = self.dataset.next_batch()
                   
                    feed_dict={input_data_placeholder: batch_data, model["labels"]: batch_labels, model["keep"]:self.hparams.keep, self.dict_model["istraining"]:True}

                    _, batch_loss, batch_acc, step = self.sess.run(
                    [model["optimizer"], model["loss"], model["metrics"]["accuracy"], model["global_step"]],
                    feed_dict=feed_dict)
                    sum_acc += batch_acc * self.hparams.batch_size
                    
                duration = time() - start_time
                mean_acc = sum_acc / (steps_per_epoch * self.hparams.batch_size)
                print(msg.format(step,  mean_acc*100, batch_loss, (steps_per_epoch / duration), 
                                 (steps_per_epoch*self.hparams.batch_size / duration) 
                                ))
                
                if epoch % 10 == 0:
                    acc = self._evaluate( val_data, val_labels )

                    print(acc, best_accuracy)
                    if acc > best_accuracy:
                        self._saver.save(self.sess, ckpt_path,  global_step=step)
                        if best_accuracy  > 0:
                            print("The model was saved. The current evaluation accuracy ({:.2f}%) is greater than the last one saved({:.2f}%).".format(acc*100,best_accuracy*100) )
                        else:
                            print("The model was saved.")

                        best_accuracy= acc
    
    def train(self):
        if self.hparams is None:
            raise Exception("Hyperparameters object was not supplied.")
        if not self.built or not self._istraining:
            self.build(istraining = True)
        
        
#         train_data, train_labels = utils.data_augmentation(train_data, train_labels)

        model = self.dict_model
        if not self.dataset.istfrecord():
            _, _ = self.dataset. get_train()
            
        steps_per_epoch = int(math.ceil(self.dataset.get_size()[0] /  self.hparams.batch_size))
        
       
        if self.sess is None:
            self.sess, self._saver = self.create_monitored_session(model, steps_per_epoch)

        
        
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:


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
            
        
        self._run_optmizer_training( model, input_data_placeholder, steps_per_epoch)
#         self.plot_data()
            

        
    def _evaluate(self, data, labels, batch_size = 96):
        
        if len(data) == 0: return
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            input_data_placeholder = self.dict_model["bottleneck_input"]
        else:
            input_data_placeholder = self.dict_model["images"]

        if self.sess is None:
            self.sess = self.create_monitored_session(self.dict_model,steps_per_epoch)

        num_data = data.shape[0]
        size = num_data//batch_size
        indices = list(range(num_data))
        global_acc = 0.0
    
        for i in range(size+1):

            begin = i*batch_size
            end = (i+1)*batch_size
            end = num_data if end >= num_data else end
             
            next_bach_indices = indices[begin:end]
            if len(next_bach_indices) == 0:
                break;
                
            batch_data = data[next_bach_indices]
            batch_labels = labels[next_bach_indices]
            

            acc = self.sess.run( self.dict_model["metrics"]["accuracy"],
                feed_dict={input_data_placeholder: batch_data, self.dict_model["labels"]: batch_labels, self.dict_model["keep"]:1.0, self.dict_model["istraining"]:False})
            
           
            global_acc += (acc * len(next_bach_indices))
 
        mes = "--> Evaluation accuracy: {:.2f}%"
        global_acc /= num_data
        print(mes.format(global_acc * 100))
       
        return global_acc


    def test_prediction(self, batch_size = 96):
        if self._istraining or not self.built:
            self.dataset.set_tfrecord(False)
            self.build(istraining = False)
        logits = self._layers[-1].output
        data, labels = self.dataset.get_test(hot=False)
        print(data.shape, labels.shape)
            
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            input_data_placeholder = self.dict_model["bottleneck_input"]
        else:
            input_data_placeholder = self.dict_model["images"]
        model = self.dict_model
        
        predictions = {
                       "classes":[],
                       "probs":[],
                       "labels":[]
                      }
        num_data = data.shape[0]
        size = num_data//batch_size
        indices = list(range(num_data))
        probs = tf.nn.softmax(logits)
        for i in range(size+1):

            begin = i*batch_size
            end = (i+1)*batch_size
            end = num_data if end >= num_data else end

            next_bach_indices = indices[begin:end]
            batch_xs = data[next_bach_indices]
            batch_ys = labels[next_bach_indices]
            if self.hparams.data_augmentation and not self.hparams.auto_crop == (0,0): 
                prob = self._pred_with_crop(input_data_placeholder, batch_xs, 
                                            self.hparams.crop[0], self.hparams.crop[1])
            else:
                prob = self.sess.run(probs,
                    feed_dict={input_data_placeholder: batch_xs, self.dict_model["keep"]:1.0,
                               self.dict_model["istraining"]:False})

            predictions["classes"].extend(np.argmax(prob,1))
            predictions["probs"].extend(prob)
            
            predictions["labels"].extend(batch_ys)
            

       
        correct =  np.array(predictions["labels"]) == np.array(predictions["classes"])
       
        acc = np.mean(correct) *100.0
        p,r,f,s = sklm.precision_recall_fscore_support(predictions["labels"],  predictions["classes"]) 
#         mes = "--> Accuracy: {:.2f}% ({}/{})"
#         print(mes.format( acc, sum(correct), len(data)))
        mes = "--> Accuracy: {:.2f}\n--> Precision: {}\n--> Recall:    {}\n--> F1-score:  {}\n\n"
        print(mes.format(acc, ["{:.2f}".format(v*100) for v in p],["{:.2f}".format(v*100) for v in r],["{:.2f}".format(v*100) for v in f]))
        

        return predictions

    def _pred_with_crop(self, input_data_placeholder, batch_xs, batch_y, crop_h,crop_w):
        logits = self._layers[-1].output
        batchs = [batch_xs[:,:-crop_w,:-batch_xs,:], batch_xs[:,crop_w:,:-batch_xs,:], batch_xs[:,crop_w:,crop_h:,:], batch_xs[:,:-crop_w,crop_h:,:]] 
        preds = []
        for batch in batchs:
            pred = self.sess.run(logits,
                feed_dict={input_data_placeholder: batch,self.dict_model["keep"]:1.0, self.dict_model["istraining"]:False})
            preds.append(pred)
        preds = np.array(preds)

        sum_logits = np.sum(preds,0, keepdims = False)
        prob = self.sess.run(tf.nn.softmax(sum_logits),1)
        return prob

