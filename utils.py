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
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import os,cv2
from scipy.misc import imread, imresize
from tensorflow.python.framework import graph_util

def plot_images(images, subplot = (1,2), show_size=100):
    if not images or len(images) == 0:
        return
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """
    from skimage.transform import resize
    def normalize_image(x):
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min)
        return x_norm

    # Create figure with sub-plots.

    fig, axes = plt.subplots(*subplot)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    size = len(images)
    for i, ax in enumerate(axes.flat):
        if i >= size: break
        # Get the i'th image and only use the desired pixels.
        img = images[i]
        img = resize(img, (show_size, show_size), anti_aliasing=True)


        # Normalize the image so its pixels are between 0.0 and 1.0
        img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
  
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
   
# https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py"
def guided_BP(sess, model, image, label_id = -1):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        label_vector = model["labels"]
        input_image = model["images"]

        logits = model["layers"]["logits"].output
        cost = logits*label_vector

        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, input_image)[0]

        init = tf.global_variables_initializer()

 
    output = [0.0]* logitas.get_shape().as_list()[1] #one-hot embedding for desired class activations
    output = np.array(output)
    prob = tf.nn.softmax(logits)
    if label_id == -1:
        prob = sess.run(prob, feed_dict={input_image:image, label_vector: output.reshape((1,-1)) })
        index = np.argmax(prob)
        print "Predicted_class: ", index
        output[index] = 1.0

    else:
        output[label_id] = 1.0

    gb_grad_value = sess.run(gb_grad, feed_dict={input_image:image, label_vector: output.reshape((1,-1))})

    return gb_grad_value[0] 

def grad_CAM_plus(sess, image, model, label_id, output_filename):
    g = tf.get_default_graph()
    hparams = model.hparams



    #define your tensor placeholders for, labels and images
    label_vector = model["labels"]
    input_image = model["images"]
    label_index = tf.placeholder("int64", ())
    logits = model["layers"]["logits"].output


    
    #prob = tf.placeholder("float", [None, 1000])

    #get the output neuron corresponding to the class of interest (label_id)
    cost = logits * label_vector

    # Get last convolutional layer gradients for generating gradCAM++ visualization
    target_conv_layer = model["layers"]["conv5/conv3_3"].output
    target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]

    #first_derivative
    first_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad

    #second_derivative
    second_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad 

    #triple_derivative
    triple_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad*target_conv_layer_grad  



    output = [0.0]*logits.get_shape().as_list()[1] #one-hot embedding for desired class activations
        #creating the output vector for the respective class
    prob = tf.nn.softmax(logits)
    output = np.array(output)
    if label_id == -1:
        prob_val = sess.run(prob, feed_dict={input_image: image, label_vector:output.reshape((1,-1))})
        index = np.argmax(prob_val)
        orig_score = prob_val[0][index]
        print "Predicted_class: ", index
        output[index] = 1.0
        label_id = index
    else:
        output[label_id] = 1.0
    
 
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run([target_conv_layer, first_derivative, second_derivative, triple_derivative], feed_dict={input_image:image, label_index:label_id, label_vector: output.reshape((1,-1))})

    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)
    #normalizing the alphas
    """	
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
    """

    alphas_thresholding = np.where(weights, alphas, 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))


    alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))



    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0   

    cam = resize(cam, (224,224))
    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0    
    cam = resize(cam, (224,224))


    gb = guided_BP(sess, model,[img1], label_id)
    visualize(img1, cam, output_filename, gb) 
    return cam

def visualize(img, cam, filename,gb_viz):
    gb_viz = np.dstack((
            gb_viz[:, :, 2],
            gb_viz[:, :, 1],
            gb_viz[:, :, 0],
        ))

    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
  
    fig, ax = plt.subplots(nrows=1,ncols=3)

    plt.subplot(141)
    plt.axis("off")
    imgplot = plt.imshow(img)

    plt.subplot(142)
    gd_img = gb_viz*np.minimum(0.25,cam).reshape(224,224,1)
    x = gd_img
    x = np.squeeze(x)
    
    #normalize tensor
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
   
    x = np.clip(x, 0, 255).astype('uint8')

    plt.axis("off")
    imgplot = plt.imshow(x, vmin = 0, vmax = 20)

    cam = (cam*-1.0) + 1.0
    cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET))
    plt.subplot(143)
    plt.axis("off")

    imgplot = plt.imshow(cam_heatmap)

    plt.subplot(144)
    plt.axis("off")
    
    cam_heatmap = cam_heatmap/255.0

    fin = (img*0.7) + (cam_heatmap*0.3)
    imgplot = plt.imshow(fin)

    plt.savefig("output/" + filename, dpi=600)
    plt.close(fig)