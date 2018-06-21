from abc import ABCMeta, abstractmethod 
import pickle
import numpy as np
import os
import sys
import tensorflow as tf
import numpy as np
import os.path
from basemodel import BaseModel
from  hyperparameters import Hyperparameters
from itertools import izip as zip
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm






class Vgg16(BaseModel):
    def __init__(self,hparams, dataset):

        BaseModel.__init__(self, hparams, 
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "fc8/BiasAdd",
                                    "file_name" : "vgg_16.ckpt",
                                    "url" : "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                                    "model_name" : "vgg_16"   
                                    },
                       dataset = dataset
                      )
        

    
    def _get_cnn(self, inputs, batch_norm = False, pretrained_weights = None): 
        net = self.repeat(2, self.conv2d, inputs, 64, [3, 3], "conv1",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool1")
        if self.hparams.cut_layer == "pool1": return net
        
        net = self.repeat(2, self.conv2d, net, 128, [3, 3], "conv2",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool2")
        if self.hparams.cut_layer == "pool2": return net
        
        net = self.repeat(3, self.conv2d, net, 256, [3, 3], "conv3",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool3")
        if self.hparams.cut_layer == "pool3": return net

        net = self.repeat(3, self.conv2d, net, 512, [3, 3], "conv4",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool4")
        if self.hparams.cut_layer == "pool4": return net

        net = self.repeat(3, self.conv2d, net, 512, [3, 3], "conv5",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool5")
        
        return net
    
    def _get_fully_connected(self, cnn_output, batch_norm = False):
        fc = self.fully_connected( input = cnn_output, hidden_units = 4096, name = "fc6")
        fc = self.fully_connected( input = fc, hidden_units = 4096, name = "fc7")
        logits = self.fully_connected( input = fc, hidden_units = 1000, name = "fc8", activation = None, batch_norm = False)
        return logits
        
        
        
class Vgg19(BaseModel):
    def __init__(self,hparams, dataset):

        BaseModel.__init__(self, hparams, 
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "fc8/BiasAdd",
                                    "file_name" : "vgg_19.ckpt",
                                    "url" : "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz",
                                    "model_name" : "vgg_19"   
                                    },
                           dataset = dataset
                      )


    def _get_cnn(self, inputs, batch_norm = False, pretrained_weights = None): 

        net = self.repeat(2, self.conv2d, inputs, 64, [3, 3], "conv1",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool1")
        if self.hparams.cut_layer == "pool1": return net

        net = self.repeat(2, self.conv2d, net, 128, [3, 3], "conv2",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool2")
        if self.hparams.cut_layer == "pool2": return net


        net = self.repeat(4, self.conv2d, net, 256, [3, 3], "conv3",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool3")
        if self.hparams.cut_layer == "pool3": return net

        
        net = self.repeat(4, self.conv2d, net, 512, [3, 3], "conv4",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool4")
        if self.hparams.cut_layer == "pool4": return net

        net = self.repeat(4, self.conv2d, net, 512, [3, 3], "conv5",  batch_norm)
        net = self.maxpool(tensor = net, name = "pool5")
        return net

    def _get_fully_connected(self, cnn_output, batch_norm = False):
        fc = self.fully_connected( input = cnn_output, hidden_units = 4096, name = "fc6")
        fc = self.fully_connected( input = fc, hidden_units = 4096, name = "fc7")
        logits = self.fully_connected( input = fc, hidden_units = 1000, name = "fc8", activation = None, batch_norm = False)
        return logits


class Inception(BaseModel):
    pass



class Resnet(BaseModel):
    pass



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


    def _get_cnn(self, inputs, batch_norm = False):
        net = self.conv2d(inputs, 32, [5, 5], name='conv1', batch_norm = batch_norm)
        net = self.max_pool2d(net, [2, 2], 2, name='pool1',  batch_norm = batch_norm)
        net = self.conv2d(net, 64, [5, 5], name='conv2',  batch_norm = batch_norm)
        net = self.max_pool2d(net, [2, 2], 2, name='pool2',  batch_norm = batch_norm)
        return net


class Mobilenet(BaseModel):
    pass



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
                net = self.conv2d(net, units, list(self._kernel_conv),  name = "conv{}".format(i),  batch_norm = False, 
                                  padding = "SAME")
                
                
        else:
            net = self._input_data = tf.layers.flatten(self._input_data)
#             net  = self.fully_connected(input=net, hidden_units=layers[0], activation = None, name = "fc_input",  
#                                        batch_norm = batch_norm)
            for i, units in zip(range(len(layers)), layers):
                net = self.fully_connected(input=net, hidden_units=units, activation = "sigmoid", name = "fc{}".format(i),
                                           batch_norm = batch_norm)
            
            net = self.fully_connected(input=net, hidden_units=3, activation = "sigmoid", name = "fc_silent",  
                                       batch_norm = batch_norm)
            
            self._silent = net
           
            for i, units in zip(range(len(layers), 2*len(layers)), layers[::-1]):
                net = self.fully_connected(input=net, hidden_units=units, activation = "sigmoid", name = "fc{}".format(i),
                                           batch_norm = batch_norm)
                
       
            
            net = self.fully_connected(input=net, hidden_units=self._input_data.shape.as_list()[-1] , 
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
        for i in range(self.hparams.num_classes):
            ax.scatter(results[labels==i, 0], results[labels==i, 1], results[labels==i, 2],
                        color=cm.get_cmap("tab20").colors[i], label=str(i), alpha=0.5)
        plt.legend()
        plt.show()
         
        return global_acc
        
        
    def _get_fully_connected(self, cnn_output, batch_norm = False): pass
    
    
    def generate_optimization_parameters(self,logits):
        
        with tf.device("cpu:0"):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.hparams.initial_learning_rate, global_step, 
                                               self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)  
        with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.squared_difference(logits, self._input_data))
                if self.hparams.regularizer_type:
                    loss = self.regularize(loss, self.hparams.regularizer_type, self.hparams.regularizer_scale)
                tf.summary.scalar("loss", loss)

        with tf.name_scope('sgd'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

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
                    mean_loss = sum_loss / (steps_per_epoch * self.hparams.batch_size*50)
                    
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
                    
       


class Ensamble(BaseModel):
    def __init__(self, hparams, dataset, description):
        hparams.cut_layer = ""
        BaseModel.__init__(self, hparams, 
                       info = {
                                    "input_images": "images", 
                                    "last_layer" : "fc8/BiasAdd",
                                    "file_name" : "ensamble",
                                    "url" : "ensamble",
                                    "model_name" : "ensamble"   
                                    },
                           dataset = dataset  
                      )
        self.description = description
        self.ckpts = []

    
    def _new_model(self, inputs,  desc_model):
        model = desc_model["model"]
        hp = Hyperparameters()
        hp.fine_tunning = True
        hp.cut_layer = desc_model["cut_layer"]
        model = model(hp, self.dataset)
        cnn = model._build_only_cnn(inputs)
        self.ckpts.append({"name":model.info["model_name"], "ckpt":model.info["file_name"], 
                           "vars_to_restore":model.vars_to_restore})

        return tf.layers.flatten(cnn), model.get_name()
        
                
    def _get_cnn(self, inputs, batch_norm = False, pretrained_weights = None): 
            independent_inputs = False
            sliced_inputs  = []
            models = self.description["models"]
            ensamble_type = self.description["type"]
            output_layers = []
            if "independent_inputs" in self.description:
                independent_inputs = self.description["independent_inputs"]

            if ensamble_type == "decaf":
                print "******************************"
                print "******* DeCaf Ensamble *******" 
                print "******************************"
                
                channels = self.hparams.channels // len(models)
                if channels == 0:
                    raise Exception("The division of the number of models by channels of data need to be different of zero!" )
                
                self.info["model_name"] += "_DeCaf"
                models.sort(key=lambda dict:dict["cut_layer"],reverse=True)
                self.info["model_name"]
                for i in range(len(models)):
                    model = models[i]
                    input = inputs if not independent_inputs else  tf.slice(inputs,[0,0,0,i*channels],
                                                                      [-1,
                                                                       self.hparams.height, self.hparams.width,3]
                                                                     )
                    flatten, name = self._new_model(input, model)
                    self.info["model_name"] += "_{}-{}".format(name, model["cut_layer"])

                    output_layers.append(flatten)
                
                ensamble = tf.concat(output_layers, 1)  #DeCaf
            
            elif ensamble_type == "sum" or ensamble_type == "avg":
                print "******************************"
                print "******** {} Ensamble ********".format( ensamble_type.upper())
                print "******************************"
                
                num_models =  self.description["num_models"]
                channels = self.hparams.channels // num_models
                if channels == 0:
                    raise Exception("The division of the number of models by channels of data need to be different of zero!" )
  
                name = "nothing"
                for i in range(num_models):
                    model = models[i]
                    input = inputs if not independent_inputs else  tf.slice(inputs,[0,0,0,i*channels],
                                                                      [-1,
                                                                       self.hparams.height, self.hparams.width, 3]
                                                                     )
                    flatten, name = self._new_model(input, model)
                    output_layers.append(flatten)

                ensamble = tf.add_n(output_layers) #SUM
                if  ensamble_type == "avg":
                    ensamble /= num_models #AVG
                 
                self.info["model_name"] += "_{}_of_{}_{}-{}".format(ensamble_type.upper(), num_models, 
                                                                    name, models[0]["cut_layer"])
           
            else:
                raise Exception("The ensable waits for a valid type (decaf, sum or avg), not {}".format(ensamble_type) )
            
            return ensamble
                  

        

    def _get_fully_connected(self, cnn_output, batch_norm = False):
        fc = self.fully_connected( input = cnn_output, hidden_units = 4096, name = "fc1")
        fc = self.fully_connected( input = fc, hidden_units = 4096, name = "fc2")
        logits = self.fully_connected( input = fc, hidden_units = 1000, name = "fc1", activation = None, batch_norm = False)
        return logits
    
    def _transfer_learning(self, inputs, only_cnn = False):
            cnn = self._get_cnn(inputs)

            if not self.hparams.fine_tunning:
                print("Stopping gradient")
                cnn = tf.stop_gradient(cnn) #Just use it in case of transfer learning without fine tunning

            flatten = tf.layers.flatten(cnn, name="flatten")
            
            if  only_cnn:
                cnn = flatten
            else:
                cnn = self._get_fully_connected(flatten)
              
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
                    print(self._input_data )
                    self.dict_model["images"] = self._input_data 
            
            dropout_keep  =  tf.placeholder(tf.float32, name='dropout_keep')
            self.dict_model["keep"] = dropout_keep

               # Create a VGG16 model and reuse its weights.
            model  = self._transfer_learning(self._input_data, only_cnn = new_fc) if  is_tf else self._from_scratch(self._input_data,
                                                                                                                only_cnn = new_fc)
                       
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
            try:
                print "Restoring the entire model."
                saver.restore(sess, tf.train.latest_checkpoint(self.hparams.checkpoints_path))
            except:
                print("It was not possible to restore the model. All variables will be initialized with their default values.")
                sess.run(tf.initialize_variables(all_vars))
            
             
           
        else:
            
            restored = []
            to_restore = []
            for ckpt in self.ckpts:
                to_restore =  [var for var in all_vars if var.name in ckpt["vars_to_restore"] ]
          
                res = tf.train.Saver(to_restore)

                try:

                    model_path = "./models/{}".format(ckpt["ckpt"])
                    print("Restoring variables from: {}".format(model_path))
                    res.restore(sess, model_path)

                except:
                    to_restore_names = ["/".join(var.name.split("/")[1:]) for var in to_restore]
                    alread_restored = [var 
                                     for var in restored 
                                     if var.name.split("/")[0]  == ckpt["name"] 
                                     and "/".join(var.name.split("/")[1:]) in  to_restore_names
                                    ]
                   
                    to_restore.sort(key=lambda var:var.name)
                    alread_restored.sort(key=lambda var:var.name)
                    
                    if len(to_restore) != len(alread_restored):
                        print("Some variables could not be restored - (Amount={})".format(len(to_restore) - len(alread_restored)))
                       
                              

             
#                     print("\n")
#                     print (len(to_restore),len(alread_restored))
            
#                     print([var.name for var in to_restore])
#                     print("\n")
#                     print([var.name for var in alread_restored])
                                     

                    for to, alread in zip(to_restore,alread_restored):
                        sess.run(to.assign(alread))

                restored += to_restore
                
                          
                    
            restored_names = [var.name for var in restored]        
            unrestored = [var for var in all_vars if var.name not in restored_names]                    
            print("Initializing only unrestored variables")
            sess.run(tf.initialize_variables(unrestored))
            
           
            
            

           
            
                  
        
        return sess, saver
    
        
        
    
    

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
        net = self.conv2d(inputs, 64, [11, 11], 4, padding='VALID', name='conv1')
        net = self.maxpool(net, [3, 3], [2,2], name='pool1')
        net = self.conv2d(net, 192, [5, 5], name='conv2')
        net = self.maxpool(net, [3, 3], 2, name='pool2')
        net = self.conv2d(net, 384, [3, 3], name='conv3')
        net = self.conv2d(net, 384, [3, 3], name='conv4')
        net = self.conv2d(net, 256, [3, 3], name='conv5')
        net = self.maxpool(net, [3, 3], 2, name='pool5')
        return net

        
        
        
        

        