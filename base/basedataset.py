from abc import ABCMeta, abstractmethod 
import pickle
import numpy as np
from random import shuffle
import random
import os
import tensorflow as tf
import time
import skimage as sk





class BaseDataset(object):
    __metaclass__ = ABCMeta
    def __init__(self, hparams,name, total_train, total_validation, total_test, tfrecord = False):
        self._generate_bottleneack = False
        self._name = name
        self.hparams = hparams
        self.total_train = total_train
        self.total_test = total_test
        self.total_validation = total_validation
        
        self._train_data = np.array([])
        self._test_data = np.array([])
        self._validation_data = np.array([])

        self._train_labels = np.array([])
        self._test_labels = np.array([])
        self._validation_labels = np.array([])

        self._indices = []
        self._next_step = 0
        self._current_epoch = 0
        self._tfrecord = tfrecord
        self._teste_sample = []
    



    #     This method needs to fill the variables
    #     _train_data, _test_data, _validation_data, 
    #     _train_labels and _validation_labels 
    #     with a numpy.array data type.
    @abstractmethod
    def load(self, name  = "train"):  pass
    
    def istfrecord(self):
        return self._tfrecord
    def set_tfrecord(self, tfrercord):
        
        self._tfrecord = tfrercord
    
    def get_name(self):
        return self._name
    
    def get_size(self):

        return ( self.total_train, self.total_test, self.total_validation)
    
    def _set_data(self, data, labels, name="train"):
        
        if name == "test":
            self._test_data, self._test_labels = data, labels
        elif name == "validation":
            self._validation_data, self._validation_labels = data, labels
        else:
            self._train_data, self._train_labels = data, labels
        
        
    def get_sample(self):
        if len(self._train_data) == 0:
            print("Loading test data...")
            self.load("test")
        if self._teste_sample:
            return self._teste_sample 
        for i in range(self.hparams.num_classes):
            for index,label in enumerate(self._test_labels):
                if label == i:
                    self._teste_sample.append(self._test_data[index])
                    break
        return self._teste_sample

    def load_pickle(self, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except Exception as e:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data
        
    def get_image_size(self):
        return self._image_size
    
    def get_current_epoch(self):
        return self._current_epoch
        
    def get_train(self, hot = False):
        if len(self._train_data) == 0:
            print("Loading train data...")
            self.load("train")
            if self.hparams.data_augmentation:
                self._train_data, self._train_labels =  self.data_augmentation(self._train_data, self._train_labels)
                self.total_train =  self._train_data.shape[0]
       
        return self._train_data, self._dense_to_one_hot(self._train_labels) if hot else self._train_labels
    
    def get_test(self, hot = False): 
        if len(self._test_data) == 0:
            print("Loading test data...")
            self.load("test")
        return self._test_data, self._dense_to_one_hot(self._test_labels) if hot else self._test_labels
    
    def get_validation(self, hot = False): 
        if len(self._validation_data) == 0:
            print("Loading validation data...")
            self.load("validation")
        return self._validation_data, self._dense_to_one_hot(self._validation_labels) if hot else  self._validation_labels
    
    def get_tfrecord_data(self, run_type = "train"):
        filename = self._generate_tfrecord_file(run_type)
        shuffled = False if type=="test" or type == "validation" else True
        return self._load_inputs(filename, shuffled )
        
    def next_batch(self, hot = False): 
        if self._train_data.shape[0] < 1 or self._train_labels.shape[0] < 1:
            print("Loading train data...")
            self.load("train")
            if self.hparams.data_augmentation:
                self._train_data, self._train_labels =  self.data_augmentation(self._train_data, self._train_labels)
                self.total_train =  self._train_data.shape[0]
       
        
        if self._indices == []:
            self._indices = list(range(self._train_data.shape[0]))
            shuffle(self._indices)
        
        begin = self._next_step * self.hparams.batch_size 
        end = begin + self.hparams.batch_size 
        
        if end > self._train_data.shape[0]:
            shuffle(self._indices)
            self._current_epoch += 1
            begin = 0
            end = self.hparams.batch_size 
            self._next_step = 1
        else:
            self._next_step += 1
        
        next_batch_indices = self._indices[begin:end]
        data = self._train_data[next_batch_indices]
        labels = np.array(self._train_labels[next_batch_indices])
        
        return data, self._dense_to_one_hot(labels) if hot else labels
    
    def data_augmentation(self, images, labels):
        
        def random_rotation(image_array):
          # pick a random degree of rotation between 25% on the left and 25% on the right
            random_degree = random.uniform(-15, 15)
            return sk.transform.rotate(image_array, random_degree)

        def random_noise(image_array):
          # add random noise to the image
            return sk.util.random_noise(image_array)

        def horizontal_flip(image_array):
          # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
            return image_array[:, ::-1]
        
        print("Augmenting data...")
        aug_images = []
        aug_labels = []

        aug_images.extend( list(map(random_rotation, images)) )
        aug_labels.extend(labels)
        aug_images.extend( list(map(random_noise,    images)) )
        aug_labels.extend(labels)
        aug_images.extend( list(map(horizontal_flip, images)) )
        aug_labels.extend(labels)
        aug_labels.extend(labels)



        return np.concatenate([np.array(aug_images), images]), np.array(aug_labels)
            
    def get_or_generate_bottleneck( self, sess, model, file_name, type = "train", batch_size = 128):

        if type == "test":
            dataset, labels = self.get_test()
            
        elif type == "validation":
            dataset, labels = self.get_validation()
        else:
            dataset, labels = self.get_train()
        
#         labels = self._dense_to_one_hot(labels)
        path_file = os.path.join("../data",file_name+".pkl")
        
        if(os.path.exists(path_file)):
            print("Loading bottleneck from \"{}\" ".format(path_file))
            with open(path_file, 'rb') as f:
                bottleneck = pickle.load(f)
       
        else:

            bottleneck_data = []

            print("Generating Bottleneck \"{}.pkl\" ".format(file_name) )
            count = 0
            amount = len(labels) // batch_size
            indices = list(range(len(labels)))
            shuffle(indices)
            for i in range(amount+1):

                if (i+1)*batch_size < len(indices):
                    indices_next_batch = indices[i*batch_size: (i+1)*batch_size]
                else:
                    indices_next_batch = indices[i*batch_size:]
                batch_size = len(indices_next_batch)

                data = dataset[indices_next_batch]
                label = labels[indices_next_batch]
                
                input_size = np.prod(model["bottleneck_tensor"].shape.as_list()[1:])
                tensor = sess.run(model["bottleneck_tensor"], 
                                  feed_dict={
                                  model["images"]:data, 
                                  model["bottleneck_input"]:np.zeros((batch_size,input_size)),  
#                                   model["labels"]:label,
#                                   model["keep"]:1.0
                                  } 
                                 )
                
                for t in range(batch_size):
                    bottleneck_data.append(np.squeeze(tensor[t]))
            
            del dataset
            bottleneck = {
            "data":np.array(bottleneck_data),
            "labels":np.array(labels)
            } 
            
            del  bottleneck_data, labels
            

            if not os.path.exists("../data"):
                os.makedirs("../data")
                
            with open(path_file, 'wb') as f:
                pickle.dump(bottleneck, f)
            print("saving")
        if type == "test":
            self._test_data = np.array(bottleneck["data"])
            self._test_labels = np.array(bottleneck["labels"])
        elif  type == "validation":
            self._validation_data = np.array(bottleneck["data"])
            self._validation_labels = np.array(bottleneck["labels"])
        else:
            self._train_data = np.array(bottleneck["data"])
            self._train_labels = np.array(bottleneck["labels"])
        del bottleneck
        print("Done")
        

    def _dense_to_one_hot(self, labels_dense):
        labels_one_hot = labels_dense
        if len(labels_dense.shape) == 1:
               labels_one_hot = np.eye(self.hparams.num_classes)[labels_dense.astype(int)]
        return labels_one_hot
    
    
    def _generate_tfrecord_file(self, run_type = "train"):
        file_name = "./data/tfrecords/{}_{}.tfrecords".format( type(self).__name__ , run_type)
        if os.path.exists(file_name):
            print( "File {}_{}.tfrecords alread exists.".format(type(self).__name__,run_type) )
            return file_name
        
        def byte_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def int_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        print("Generating tfrecord file: {}".format(file_name))
        with tf.device('/cpu:0'):
    #     key, value = reader.read(filename_queue)
    #     decoded_image = tf.image.decode_png(value) # tf.image.decode_image
    #     decoded_image = tf.reshape(tf.cast(decoded_image, tf.uint8), [-1])
    
            t = time.time()
            count = 0
            config = tf.ConfigProto()
            config.intra_op_parallelism_threads = 8
            config.inter_op_parallelism_threads = 8
            writer = tf.python_io.TFRecordWriter(file_name)

            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                if type == "test":
                    data_set, labels = self.get_test(hot = False)
                elif type == "validation":
                    data_set, labels = self.get_validation(hot = False)
                else:
                    data_set, labels = self.get_train(hot = False)
                
                indices = list(range(len(data_set)))
                shuffle(indices)
                for index in indices:
                    data = data_set[index].astype(np.uint8)
                    label = labels[index]


                    if label == None:
                        continue

                    example = tf.train.Example(features=tf.train.Features(feature={       
                        'data': byte_feature(data.tostring()), 
                        'label':int_feature([label])
                    }))
                    writer.write(example.SerializeToString())
                    if count % 500 == 0:
                        print('Round:', count)

                    count += 1
                
                del data_set, labels
                writer.close()
                coord.request_stop()
                coord.join(threads)
            print('Elapsed time:',time.time() - t)
            
            
        return file_name
    
    def _tf_data_aug(self, data):
        print("Data augmentation enabled!")
        data = tf.random_crop(data, [self.hparams.height-self.hparams.auto_crop[0], 
                                      self.hparams.width-self.hparams.auto_crop[1],
                                      self.hparams.channels])
        data = tf.image.random_flip_left_right(data)

        return data
        
    def _read_decode_distort(self,queue,  is_train = True):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(queue)
        features = tf.parse_single_example(serialized, features={
        'data': tf.FixedLenFeature([self.hparams.height *
                                         self.hparams.width *
                                         self.hparams.channels ], 
                                        tf.string),
            
        'label': tf.FixedLenFeature([], tf.int64),
        })
        
        data = tf.decode_raw(features['data'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        
        data = tf.cast(data, tf.float32)
        data = tf.reshape(data, [self.hparams.height, self.hparams.width, self.hparams.channels])
        if self.hparams.data_augmentation and is_train: 
            
            data = self._tf_data_aug(data)


        else:
            
            data = tf.squeeze(tf.image.resize_bicubic(tf.expand_dims(data,0), 
                                                      [self.hparams.height-self.hparams.auto_crop[0],
                                                       self.hparams.width-self.hparams.auto_crop[1]]
                                                     )
                             )
        return data, label

    def _load_inputs(self, filename, shuffled = True):
        
        queue = tf.train.string_input_producer([filename])

        data, label = self._read_decode_distort(queue, shuffled )
        if shuffled:
            data_batch, label_batch = tf.train.shuffle_batch(
                [data, label], 
                batch_size = self.hparams.batch_size, 
                num_threads =11, 
                capacity = int(self.total_train * 0.05) + 3 * self.hparams.batch_size, 
                min_after_dequeue = int(self.total_train * 0.05))
        else:
            data_batch, label_batch = tf.train.batch(
                [data, label], 
                batch_size = self.hparams.batch_size, 
                num_threads = 11, 
                capacity = int(self.total_validation * 0.4))

#         label_batch = tf.one_hot(label_batch, self.hparams.num_classes)
        return data_batch, label_batch
    
    
    
    

