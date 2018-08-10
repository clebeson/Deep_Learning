from base import BaseDataset
import utils
import numpy as np
import os
import pickle


class CE(BaseDataset):
    
    def __init__(self, hparams, tfrecord = False):
            hparams.num_classes = 7
            hparams.height = 100
            hparams.width = 100
            hparams.channels = 3
            hparams.class_names = ["neutro", "feliz", "triste", "medo", "raiva", "surpresa", "nojo"]
            BaseDataset.__init__(self, hparams = hparams, name = "ce", total_train = 1050, total_test = 350, total_validation = 350, 
                                 tfrecord = tfrecord)
           
            

    def load(self, name="train"):
        folder_name = 'ce_database'
        main_directory = "./data"
#         url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

#         utils.maybe_download_and_extract(url, main_directory, folder_name, "cifar-10-batches-py")
# #         f = open(os.path.join(main_directory,folder_name,"batches.meta"), 'rb')
# #         f.close()
        if name == "train":
            for i in range(3):
                datadict = self.load_pickle(os.path.join(main_directory,folder_name,'data_batch_' + str(i + 1)))

           
                data = np.array(datadict["data"])
                labels = np.array(datadict['labels'])-1

                data = np.array(data, dtype=float) 
                data = data.reshape([-1, 3, 100, 100])
                data = data.transpose([0, 2, 3, 1])

                if i == 0:
                    self._train_data = data
                    self._train_labels = labels
                else:
                    self._train_data = np.concatenate((self._train_data, data), axis=0)
                    self._train_labels = np.concatenate((self._train_labels, labels), axis=0)
            
            
        elif name == "validation":
            datadict = self.load_pickle(os.path.join(main_directory,folder_name,'test_batch'))
          

            data = np.array(datadict["data"])
            data = np.array(data, dtype=float) 
            data = data.reshape([-1, 3, 100, 100])
            data = data.transpose([0, 2, 3, 1])
            labels = np.array(datadict['labels']) - 1


            self._validation_data = data
            self._validation_labels = labels
        
        else:

            datadict = self.load_pickle(os.path.join(main_directory,folder_name,'test_batch'))
     

            data = np.array(datadict["data"])
            data = np.array(data, dtype=float) 
            data = data.reshape([-1, 3, 100, 100])
            data = data.transpose([0, 2, 3, 1])
            labels = np.array(datadict['labels']) - 1


            self._test_data = data
            self._test_labels = labels


                        
        del data, labels, datadict  
                        
    def generate_channels(self, data):
        aug = []

        for d in data:
            
            aug.append(np.concatenate((d,d,d), 2))
        
        return np.array(aug)
        
                        

