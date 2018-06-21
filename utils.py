import pickle
import numpy as np
import os
# from urllib.request import urlretrieve
import urllib2
import tarfile
import zipfile
import sys
import skimage as sk
from skimage import transform
from skimage import util
import random
import math
import os.path
from random import shuffle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product


  
def maybe_download_and_extract(url, main_directory,filename, original_name):
    def _print_download_progress( count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r --> progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    if not os.path.exists(os.path.join(main_directory,filename)):
        if not os.path.exists(main_directory): os.makedirs(main_directory)
        
        url_file_name = url.split('/')[-1]
        zip_file = os.path.join(main_directory,url_file_name)
        print("Downloading ",url_file_name)

        try:
            f = urllib2.urlopen(url)
            data = f.read()
            with open(zip_file, "wb") as code:
                code.write(data)
    #                 file_path, _ = urlretrieve(url=url, filename= zip_file, reporthook=_print_download_progress)
        except:
                print("This could be for a problem with the storage site. Try again later")
                return

        print("\nDownload finished.")

        if zip_file.endswith(".zip"):
            print( "Extracting files.")
            zipfile.ZipFile(file=zip_file, mode="r").extractall(main_directory)

        elif zip_file.endswith((".tar.gz", ".tgz")):
            print( "Extracting files.")
            tarfile.open(name=zip_file, mode="r:gz").extractall(main_directory)
            os.remove(zip_file)

        os.rename(os.path.join(main_directory,original_name), os.path.join(main_directory,filename))
        print("Done.")
     
    
def data_augmentation(images, labels):

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


    return np.array(aug_images), np.array(aug_labels)
        
        
        
def generate_confusion_matrix( predictions, class_names):
        
    def plot_confusion_matrix(cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm.shape)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))


        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        symbol = "%" if normalize else ""
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt)+symbol,
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Real')
        plt.xlabel('Predicted')
        
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(predictions["labels"],predictions["classes"])
    np.set_printoptions(precision=2)


    # # Plot normalized confusion matrix
    plt.figure(figsize=(10,7))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
    plt.grid('off')

    #plt.savefig("./confusion_matrix.png") #Save the confision matrix as a .png figure.
    plt.show()