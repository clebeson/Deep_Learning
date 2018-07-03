from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import cv2
import pims
from moviepy.editor import *
import scipy.misc as m
from itertools import izip as zip
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle



def get_label(name):
        n, _ = os.path.splitext(name)
        n = int(n.split("_")[-1]) -1
        return n

def normalize(image, star):
    
    for i in range(3):
        s = star[:,:,i]
        img = image[:,:,i]
        max = np.max(s[:])
        min = np.min(s[:])
        image[:,:,i] = 255* (img-min) / (max-min) 
    return np.uint8(image)
def normalize(image):
    if len(list(image.shape)) == 3:
        for i in range(3):
            img = image[:,:,i]
            max = np.max(img[:])
            min = np.min(img[:])
            image[:,:,i] = 255* (img-min) / (max-min) 
    else:
        max = np.max(image[:])
        min = np.min(image[:])
        image = 255* (image-min) / (max-min) 

    return np.uint8(image)


def diff(img1, img2):
         return np.absolute(img1-img2)

def add(img1, img2):
    img = img1 + img2
    return img

def get_star(frames ):
    #resize and convert to float32
    return reduce(add, map(diff, frames[:-1],frames[1:]))

def get_star_1_RGB(frames):
    total = len(frames) 
    if list(frames[0].shape)[-1] == 3:
        frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*total))
    step = total //3
    r = np.expand_dims(get_star(frames[:step]), 2)
    g = np.expand_dims(get_star(frames[step:step*2]), 2)
    b = np.expand_dims(get_star(frames[step*2:]), 2)
    
    return normalize(np.concatenate([r,g,b], axis = 2))

def get_star_3_RGB(frames):
    total = len(frames)
    step = total //3
    frames_1 = list(map(lambda f: f[:,:,0], frames))
    frames_2 = list(map(lambda f: f[:,:,1], frames))
    frames_3 = list(map(lambda f: f[:,:,2], frames))
    image1 = get_star_1_RGB(frames_1)
    image2 = get_star_1_RGB(frames_1)
    image3 = get_star_1_RGB(frames_1)
    return np.concatenate([image1,image2,image3], axis = 2)
    


def get_frames(file, size = (120,160)):
    frames = pims.Video(file)
    frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
    return frames
