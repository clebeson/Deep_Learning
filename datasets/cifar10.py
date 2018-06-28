from base import BaseDataset
import utils
import numpy as np
import os
import pickle


class Cifar10(BaseDataset):
    
    def __init__(self, hparams, tfrecord = False):
            hparams.num_classes = 10
            hparams.height = 32
            hparams.width = 32
            hparams.channels = 3
            hparams.class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
            BaseDataset.__init__(self, hparams = hparams, name = "Cifar10", total_train = 45000, total_test = 10000, total_validation = 5000, 
                                 tfrecord = tfrecord)
           
            

    def load(self):
        folder_name = 'cifar_10'
        main_directory = "./data"
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        utils.maybe_download_and_extract(url, main_directory, folder_name, "cifar-10-batches-py")
#         f = open(os.path.join(main_directory,folder_name,"batches.meta"), 'rb')
#         f.close()

        for i in range(5):
            
            f = open(os.path.join(main_directory,folder_name,'data_batch_' + str(i + 1)), 'rb')
            datadict = pickle.load(f)
            f.close()

            data = np.array(datadict["data"])
            labels = datadict['labels']

            data = np.array(data, dtype=float) / 255.0
            data = data.reshape([-1, 3, 32, 32])
            data = data.transpose([0, 2, 3, 1])

            if i == 0:
                self._train_data = data
                self._train_labels = labels
            else:
                self._train_data = np.concatenate((self._train_data, data), axis=0)
                self._train_labels = np.concatenate((self._train_labels, labels), axis=0)

        f =  open(os.path.join(main_directory,folder_name,'test_batch'), 'rb')
        datadict = pickle.load(f)
        f.close()

        data = np.array(datadict["data"])
        labels = np.array(datadict['labels'])
        

        self._test_data = np.array(data, dtype=float) / 255.0
        self._test_data = self._test_data.reshape([-1, 3, 32, 32])
        self._test_data = self._test_data.transpose([0, 2, 3, 1])
        self._test_labels = labels
        
        indices = list(range(len(self._train_data)))
        index_samples  = np.random.choice(len(self._train_data), len(self._train_data) // 10, replace=False)

        indices = list(set(indices).difference(set(index_samples)))
                     
        self._validation_data = self._train_data[index_samples]
        self._train_data = self._train_data[indices]
        
#         self._test_data  = self.generate_channels(self._test_data)
#         self._validation_data = self.generate_channels(self._validation_data)
#         self._train_data = self.generate_channels( self._train_data )
        
        
        
       
        self._validation_labels = self._train_labels[index_samples]
        self._train_labels = self._train_labels[indices]
      
       
                        
        del indices, index_samples, data, labels, datadict  
                        
    def generate_channels(self, data):
        aug = []

        for d in data:
            
            aug.append(np.concatenate((d,d,d), 2))
        
        return np.array(aug)
        
                        

