import os
import random
import math
#from math import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from contextlib import redirect_stdout
from io import StringIO
from termcolor import colored
import sys
os.system('color')
import os.path
#from albumentations import ToFloat, Rotate

# normalize image into range <0;1>
def normalize_image(img):
	return (img / 255).astype('float32')

# normalize image z-score
def normalize_image_z(img, img_mean, img_std):
	return ((img - img_mean) / img_std).astype('float32')

# augment image with custom rotation (which changes image dimensions)
def augment_img(img, augmentation, rotation_limit=0, rotation_prob=0.5):
	img = augmentation(image=img)["image"]
	if rotation_limit > 0:
		if random.random() < rotation_prob:
			img = rotate_image(img, (-1.0+random.random() * 2.0) * rotation_limit)

	return img

# desplay the dataset samples
def DrawPics(tensor,nb=0,template='{}',classnumber=None):
    if (nb==0):
        N = tensor.shape[0]
    else:
        N = min(nb,tensor.shape[0])
    print(N)
    fig = plt.figure(figsize=(10,2))
    nbligne = math.floor(N/20)+1
    for m in range(N):
        subplot = fig.add_subplot(nbligne,min(N,20),m+1)
        subplot.axis("off")
        plt.imshow(tensor[m,:,:,0],vmin=0, vmax=1,cmap='Greys')
        if (classnumber!=None):
            subplot.title.set_text((template.format(classnumber)))

#######################################################################################
def drawTriplets(tripletbatch, nbmax=None):
    """display the three images for each triplets in the batch
    """
    labels = ["Anchor", "Positive", "Negative"]
    if (nbmax==None):
        nbrows = tripletbatch[0].shape[0]
    else:
        nbrows = min(nbmax,tripletbatch[0].shape[0])
    for row in range(nbrows):
        fig=plt.figure(figsize=(16,2))
        for i in range(3):
            subplot = fig.add_subplot(1,3,i+1)
            subplot.axis("off")
            plt.imshow(tripletbatch[i][row,:,:,0])
            subplot.title.set_text(labels[i])

# resizes and pads image to defined size
def preprocess_img(img, size=(256, 256), channels=1):
	# calculate new size of image
	size_old = img.shape[:2]
	ratio = min(float(size[0])/size_old[0], float(size[1])/size_old[1])
	size_new = tuple([int(x*ratio) for x in size_old])

	# resize image
	img = cv2.resize(img, (size_new[1], size_new[0]), interpolation = cv2.INTER_CUBIC)

	# pad image
	img = pad_image(img, size)
	return img
# pads image to desired size
#  - ads surrounding with matching color and gauss noise
def pad_image(img, size):
	img_shape = img.shape[:2]

	if size[0] < img_shape[0]:
		size = (img_shape[0], size[1])

	if size[1] < img_shape[1]:
		size = (size[0], img_shape[1])

	# get background color
	color = findDominantBorderColor(img)

	# get padding sizes
	dw = size[1] - img_shape[1]
	dh = size[0] - img_shape[0]
	left, right = dw//2, dw-(dw//2)
	top, bottom = dh//2, dh-(dh//2)

	# create container
	img_bottom = np.full(size, color)

	# add gauss noise
	mean = 0
	var = 0.1
	sigma = var**0.5
	gauss = np.random.normal(mean, sigma, img_bottom.shape)
	img_bottom += gauss*8

	# insert original image
	img_bottom[top:top+img_shape[0], left:left+img_shape[1]] = img

	return img_bottom
# find dominant color in an image
def findDominantColor(img):
	return np.median(img[0:int(img.size)])

# finds dominant color in image borders
def findDominantBorderColor(img):
	shape_d = img.shape[:2]
	shape = (shape_d[0]-1, shape_d[1]-1)

	border_med = 0.0
	border_med += np.median(img[0:shape[0], 0])
	border_med += np.median(img[0:shape[0], shape[1]])
	border_med += np.median(img[0, 0:shape[1]])
	border_med += np.median(img[shape[0], 0:shape[1]])

	return border_med / 4.0
# flips an image into horizontal position
def to_horizontal(image):
	h, w = image.shape[:2]

	if w < h:
		image = np.rot90(image)
	return image

# rotate image by angle, fill surrounding with dominant color
def rotate_image(img, angle):
	h, w = img.shape[:2]
	center = (w/2, h/2)

	matrix = cv2.getRotationMatrix2D(center, angle, 1.)

	abs_cos = abs(matrix[0,0])
	abs_sin = abs(matrix[0,1])

	bound_w = int(h * abs_sin + w * abs_cos)
	bound_h = int(h * abs_cos + w * abs_sin)

	matrix[0,2] += bound_w/2 - center[0]
	matrix[1,2] += bound_h/2 - center[1]

	color = findDominantBorderColor(img)
	return cv2.warpAffine(img, matrix, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=color)

# print message to stderr
def eprint(*args, **kwargs):
	with StringIO() as sstream, redirect_stdout(sstream):
		print(*("ERROR:", *args), end='')
		text = colored(sstream.getvalue(), 'red')

	print(text, file=sys.stderr, **kwargs)

# print message to stderr
def wprint(*args, **kwargs):
	with StringIO() as sstream, redirect_stdout(sstream):
		print(*("WARNING: ", *args), end='')
		text = colored(sstream.getvalue(), 'yellow')

	print(text, file=sys.stderr, **kwargs)

# print message to stdout
def oprint(*args, **kwargs):
	with StringIO() as sstream, redirect_stdout(sstream):
		print(*("SUCCESS: ", *args), end='')
		text = colored(sstream.getvalue(), 'green')

	print(text, **kwargs)