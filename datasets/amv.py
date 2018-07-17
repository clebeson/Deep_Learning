from base import BaseDataset
import utils
import numpy as np
import os
import pickle
from  itertools import izip as zip


class AMV(BaseDataset):
    
    def __init__(self, hparams, tfrecord = False):
            hparams.num_classes = 2
            hparams.height = 150
            hparams.width = 75
            hparams.channels = 3 
            hparams.class_names = ["nao","sim"]
            
            BaseDataset.__init__(self, hparams = hparams, name = "AMV", total_train = 1389, total_test =76 , total_validation = 76, 
                                 tfrecord = tfrecord)
           
            
    def concat(self,images):
        images = [np.expand_dims(image,-1) for image in images]
        return np.concatenate(images,2)
        
    def load(self, name = "train"):
        folder_name = 'amv'
        main_directory = "./data"
        


        with open(os.path.join(main_directory,
folder_name,"amv_{}.pkl".format(name if not name == "validation" else "test")), 'rb') as f:
            datadict = pickle.load(f)
        
        self._set_data(np.array([self.concat([d,d,d])/255.0 for d in datadict["data"]]), np.array(datadict['labels']), name = name)
        del datadict

#   
                        

