from base import BaseDataset
import utils
import numpy as np
import os
import pickle


class Star_cos(BaseDataset):
    
    def __init__(self, hparams, tfrecord = False):
            hparams.num_classes = 20
            hparams.height = 100
            hparams.width = 100
            hparams.channels = 3
            hparams.class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 
                                   'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi', 
                                   'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 
                                   'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo'
                                  ]
            
            BaseDataset.__init__(self, hparams = hparams, name = "Star_cos", total_train = 6847, total_test =3579 , total_validation = 2700, 
                                 tfrecord = tfrecord)
           
            

    def load(self, name = "train"):
        
        folder_name = 'star_cos'
        main_directory = "./data"
#         url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#         utils.maybe_download_and_extract(url, main_directory, folder_name, "cifar-10-batches-py")
#         f = open(os.path.join(main_directory,folder_name,"batches.meta"), 'rb')
#         f.close()


        file_name = os.path.join(main_directory,folder_name,"star_cos_{}.pkl".format(name))
        datadict = self.load_pickle(file_name)
        self._set_data(np.array([d[10:-10, 30:-30,:]/255.0 for d in datadict["data"]]), np.array(datadict['labels']), name = name)
        del datadict

#   
                        

