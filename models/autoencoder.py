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





class Autoencoder(BaseModel):
    def __init__(self,hparams, dataset, kernel_conv = None ):

        BaseModel.__init__(self, hparams, 
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "last_layer",
                                    "file_name" : "",
                                    "url" : "",
                                    "model_name" : "autoencoder"   
                                    },
                           dataset = dataset
                      )
        self._kernel_conv = kernel_conv
        self._salient = None
        


    def _get_cnn(self, inputs, batch_norm = False, pretrained_weights = None):
#         if len(self.hparams.hidden_layers) < 2:
#             raise Exception("The number of layers must have at least 2." )
        
        print("INFO: Be concious that the number of layers passed is only for the encoder. The decoder will be its symetric")
        
          
        layers = self.hparams.hidden_layers 

        if self._kernel_conv is not None:
            net = inputs  
            
            
            for i, units in zip(range(len(layers)), layers):
                net = bf.conv2d(net, units, list(self._kernel_conv),  name = "conv{}".format(i),  batch_norm = False, 
                                  padding = "SAME")
                
                
        else:
            net = self._input_data = tf.layers.flatten(self._input_data)
#             net  = self.fully_connected(input=net, hidden_units=layers[0], activation = None, name = "fc_input",  
#                                        batch_norm = batch_norm)
            for i, units in zip(range(len(layers)), layers):
                net = bf.fully_connected(input=net, hidden_units=units, activation = "sigmoid", name = "fc{}".format(i),
                                           batch_norm = batch_norm)
            
            net = bf.fully_connected(input=net, hidden_units=3, activation = "sigmoid", name = "fc_silent",  
                                       batch_norm = batch_norm)
            
            self._silent = net
           
            for i, units in zip(range(len(layers), 2*len(layers)), layers[::-1]):
                net = bf.fully_connected(input=net, hidden_units=units, activation = "sigmoid", name = "fc{}".format(i),
                                           batch_norm = batch_norm)
                
            net = bf.fully_connected(input=net, hidden_units=self._input_data.shape.as_list()[-1] , 
                                       activation = None, name = "fc_output", batch_norm = batch_norm)
   
        return net
            
                

        
    def plot_data(self, batch_size = 300):
        data, labels = self.dataset.get_train()
        if (not self.hparams.fine_tunning) and self.hparams.bottleneck:
            input_data_placeholder = self.dict_model["bottleneck_input"]
        else:
            input_data_placeholder = self.dict_model["images"]

        if self.sess is None:
            self.sess = self.create_monitored_session(self.dict_model,steps_per_epoch)

        size = len(data)//batch_size
        indices = list(range(len(data)))
        global_acc = 0;
        results = np.array([])
        for i in range(size+1):

            begin = i*batch_size
            end = (i+1)*batch_size
            end = len(data) if end >= len(data) else end
            
            next_bach_indices = indices[begin:end]
            if len(next_bach_indices) == 0:
                break;
                
            batch_data = data[next_bach_indices]
            batch_labels = labels[next_bach_indices]
            
       
            silent = self.sess.run(self._silent,
                feed_dict={input_data_placeholder: batch_data, self.dict_model["labels"]: batch_labels, 
                           self.dict_model["keep"]:1.0})
            if len(results) == 0: 
                results = silent
            else:   
                results = np.concatenate([results,silent])
        results = np.array(results)
#         c = np.random.choice(range(10), results.shape[0])
#         print(results.shape)
#         # plot the data
        print(results[:20])
        
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
# #         # plot x,y data with c as the color vector, set the line width of the markers to 0
#         ax.scatter(results[:,1], results[:,0])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        labels = np.argmax(labels,1)
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i in range(self.hparams.num_classes):
            ax.scatter(results[labels==i, 0], results[labels==i, 1], results[labels==i, 2],
                        color=colors[i], label=str(i), alpha=0.5)
        plt.legend()
        plt.show()
         
        return global_acc
        
        
    def _get_fully_connected(self, cnn_output, batch_norm = None): pass
    
    
    def generate_optimization_parameters(self,logits):
        
        with tf.device("cpu:0"):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.hparams.initial_learning_rate, global_step, 
                                               self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)  
        with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.squared_difference(logits, self._input_data))
                if self.hparams.regularizer_type:
                    loss = bf.regularize(loss, self.hparams.regularizer_type, self.hparams.regularizer_scale)
                tf.summary.scalar("loss", loss)

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.name_scope('train_accuracy'):
            acc = loss
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
        
    
    def build(self, is_tf = True, new_fc = False):
        return BaseModel.build(self, is_tf = True, new_fc = False)
    
    def _transfer_learning(self, inputs, only_cnn = False):
        return BaseModel._transfer_learning(self,inputs, only_cnn = True)
    
    def _run_optmizer_training(self, saver, model, input_data_placeholder, steps_per_epoch):
        ckpt_path = os.path.join(self._generate_checkpoint_path(),"model")
        msg = "--> Global step: {:>5} - Mean loss: {:.4f}  ({:.2f}, {:.2f}) (steps,images)/sec"
        best_accuracy = 0
        total_images = 0
        
        if  self.dataset.istfrecord():
            coord = tf.train.Coordinator() 
#             threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            
#             try:    
#                 feed_dict={model["keep"]:self.hparams.keep}
#                 sum_acc= 0
#                 for epoch in range(self.hparams.num_epochs):
#                     start_time = time()

#                     print("\n*************************************************************")
#                     print("Epoch {}/{}".format(epoch+1, self.hparams.num_epochs))

                    
#                     for s in range(steps_per_epoch):
#                         _, batch_loss, batch_acc, step = self.sess.run(
#                         [model["optimizer"], model["loss"], model["accuracy"], model["global_step"]], feed_dict=feed_dict)
#                         sum_acc += batch_acc * self.hparams.batch_size
#                     if epoch % 50 == 0:
#                         mean_acc = sum_acc / (steps_per_epoch * self.hparams.batch_size*50)
#                         print(msg.format(step,  mean_acc, batch_loss, (steps_per_epoch / duration), 
#                                          (steps_per_epoch*self.hparams.batch_size / duration) 
#                                         ))
#                         sum_acc= 0
#                         duration = time() - start_time


#             except tf.errors.OutOfRangeError:
#                 print('Done training... An error was ocurred during the training!!!')
#             finally:
#                 coord.request_stop()

#             coord.join(threads)
#             print('Done training...')
 
        
        else:
   
            train_data, train_labels = self.dataset.get_train()

            val_data, val_labels = self.dataset.get_validation()
            start_time = time()
            sum_loss= 0
            for epoch in range(self.hparams.num_epochs):
                

          
                
                for s in range(steps_per_epoch):

                    batch_data, batch_labels = self.dataset.next_batch()
                    feed_dict={input_data_placeholder: batch_data, model["labels"]: batch_labels, model["keep"]:self.hparams.keep}

                    _, batch_loss,step = self.sess.run([model["optimizer"], model["loss"],  model["global_step"]],
                                                       feed_dict=feed_dict)
                    sum_loss += batch_loss * self.hparams.batch_size
                if epoch % 50 == 0:
                    
                    duration = time() - start_time
                    mean_loss = sum_loss / (steps_per_epoch * self.hparams.batch_size * 50)
                    
                    print("\n*************************************************************")
                    print("Epoch {}/{}".format(epoch+1, self.hparams.num_epochs))
                    print(msg.format(step,  mean_loss, (steps_per_epoch * 50 / duration), 
                                     (steps_per_epoch * self.hparams.batch_size * 50 / duration) 
                                    ))
                    sum_loss= 0
                    start_time = time()


               
#                 loss = self._evaluate( val_data, val_labels )

                if epoch % 10 == 0:
                    saver.save(self.sess,ckpt_path, global_step=step)
#                     print "The model was saved."
        
        saver.save(self.sess,ckpt_path, global_step=step)
                    
        

        
        
        
