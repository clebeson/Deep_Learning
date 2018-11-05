from base import BaseDataset
import utils
import numpy as np
import os
import pickle
from random import shuffle


class StarGrit(BaseDataset):
    
    def __init__(self, hparams, is_3_images = True, tfrecord = False):
            self.num_images = 3 if is_3_images else 1
            hparams.num_classes = 9
            hparams.height = 60
            hparams.width = 80
            hparams.channels = 3 * self.num_images
            hparams.class_names = ["abort", "circle", "hello", "no", "stop", "turn", "turn_left", "turn_right", "warn"]
            
            BaseDataset.__init__(self, hparams = hparams, name = "grit_star".format(self.num_images), total_train = 434, 
                                 total_test = 109, total_validation = 0, tfrecord = tfrecord)
           
            

    def load(self, name = "train"):
        folder_name = 'star_grit'
        main_directory = "./data"
#         url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#         utils.maybe_download_and_extract(url, main_directory, folder_name, "cifar-10-batches-py")
#         f = open(os.path.join(main_directory,folder_name,"batches.meta"), 'rb')
#         f.close()

        
        with open(os.path.join(main_directory,
                              folder_name,"star_grit.pkl".format(self.num_images,name)), 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
        
        data = np.array(datadict["data"])
        labels = np.array(datadict["labels"])
        indices = list(range(543))
        shuffle(indices)
        self._set_data(data[indices[:434]]/255.0, labels[indices[:434]], name = "train")
        self._set_data(data[indices[434:]]/255.0, labels[indices[434:]], name = "test")
        del datadict, data, labels

#   
                        

