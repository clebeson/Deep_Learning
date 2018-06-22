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
                  

        

    def _get_fully_connected(self, cnn_output, batch_norm = None):
        fc = bf.fully_connected( input = cnn_output, hidden_units = 4096, name = "fc1")
        fc = bf.fully_connected( input = fc, hidden_units = 4096, name = "fc2")
        logits = bf.fully_connected( input = fc, hidden_units = 1000, name = "fc1", activation = None, batch_norm = False)
        return logits
    
    def _transfer_learning(self, inputs, only_cnn = False):
            cnn = self._get_cnn(inputs, self.haparams.batch_norm)

            if not self.hparams.fine_tunning:
                print("Stopping gradient")
                cnn = tf.stop_gradient(cnn) #Just use it in case of transfer learning without fine tunning

            flatten = tf.layers.flatten(cnn, name="flatten")
            
            if  only_cnn:
                cnn = flatten
            else:
                cnn = self._get_fully_connected(flatten,  self.haparams.batch_norm)
              
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

                logits = self.add_fully_connected(flatten,  self.haparams.batch_norm)
      
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
    
        
        
    
    

