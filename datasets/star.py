from base import BaseDataset
import utils
import numpy as np
import os
import pickle
from  itertools import izip as zip


class Star(BaseDataset):
    
    def __init__(self, hparams, is_3_images = False, tfrecord = False):
            self.num_images = 3 if is_3_images else 1
            hparams.num_classes = 20
            hparams.height = 100
            hparams.width = 100
            hparams.channels = 3 * self.num_images
            hparams.class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 
                                   'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi', 
                                   'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 
                                   'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo'
                                  ]
            
            BaseDataset.__init__(self, hparams = hparams, name = "Star_{}".format(self.num_images), total_train = 6847, total_test =3579 , total_validation = 2700, 
                                 tfrecord = tfrecord)
           
            

    def load(self, name = "train"):
        folder_name = 'star'
        main_directory = "./data"
#         url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#         utils.maybe_download_and_extract(url, main_directory, folder_name, "cifar-10-batches-py")
#         f = open(os.path.join(main_directory,folder_name,"batches.meta"), 'rb')
#         f.close()


        with open(os.path.join(main_directory,
folder_name,"star_{}_rgb_{}.pkl".format(self.num_images,name)), 'rb') as f:
            datadict = pickle.load(f)
        
        self._set_data(np.array([d[10:-10, 30:-30,:]/255.0 for d in datadict["data"]]), np.array(datadict['labels']), name = name)
        del datadict

#   
                        

