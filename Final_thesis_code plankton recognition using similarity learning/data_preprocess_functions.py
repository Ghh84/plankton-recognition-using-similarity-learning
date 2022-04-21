import cv2
import math
from math import *
import os
import random
import numpy as np
import skimage.io
from albumentations import Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightness, Blur, GaussNoise, RandomRotate90
from shutil import copyfile
from utils import augment_img,oprint, preprocess_img, normalize_image,eprint
os.system('color')
import os.path
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from os import path
import matplotlib.pyplot as plt
from PIL import Image
#from util import eprint, oprint, augment_img, preprocess_img, normalize_image_z

###########################################################################################################

'''
Rotates image by 90 degrees with adjusting image dimensions (it won't get cropped)
'''
class NetworkConfig:
	def __init__(self, min_samples=0):
		self.min_samples = min_samples
class CustomRandomRotate90(RandomRotate90):
	def get_params(self):
		return {"factor": 1}
networkConfig = NetworkConfig()
networkConfig.model_path = None
networkConfig.min_samples = 10
networkConfig.data_location = "./labeled_20201020"
networkConfig.extend = '.png'
networkConfig.prepare_dir = ''
networkConfig.prepare_num = 10
networkConfig.prepare_test_size = 20/ float(100)
output_dir="."
#networkConfig.data_location='D:\II Year LUT\Second Semister\Masters Thesis and Seminar\practice\plankton recognition practice\Plankton_Recognition'
source_dir="./train_200"
AUGMENTATIONS_TRAIN = Compose([
		ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_REPLICATE, p=0.9),
		RandomBrightness(limit=0.1, p=0.8),
		Blur(blur_limit=2, p=0.2),
		GaussNoise(0.001)
])

#img_rows, img_cols = 224,224
image_size = (224,224)
#classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
bn_classes = 14 #len(classes)
############################################################################################################
#def prepare_dataset(networkConfig, output_dir, class_balance=10, augment=True):
# 	print("-----------------------------------------------------")
# 	print("Preparing dataset")
# 	print("  Input directory: ", networkConfig.data_location)
# 	print("  Output directory: ", output_dir)
# 	print("  Samples per class: ", class_balance)
# 	print("  Augmentation: ", augment)
# 	print("  Class min samples filter: ", networkConfig.min_samples)
# 	print("  Image extension: ", networkConfig.extend)
# 	print("  Test portion: ", networkConfig.prepare_test_size)
# 	print("-----------------------------------------------------")
# 	# load classes with minimal samples
# 	classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
# 	classes_n = len(classes)
# 	print("Number of filtered classes: ", classes_n)
# 	# create test dir
# 	test_dir = os.path.join(output_dir, "test")
# 	if not os.path.exists(test_dir):
# 		os.makedirs(test_dir)
# 	# create train dir
# 	train_dir = os.path.join(output_dir, "train")
# 	if not os.path.exists(train_dir):
# 		os.makedirs(train_dir)
# 	# copy existing files
# 	print("Copying files...")
# 	for i, cl in enumerate(classes):
#         print("Processing {}/{}, class: {}".format(i+1, classes_n, cl))
#         source_dir = os.path.join(networkConfig.data_location, cl)
# 		files = sorted([f for f in os.listdir(source_dir) if f.endswith(networkConfig.extend)])
# 		# shuffle list
# 		random.Random(42).shuffle(files)
# 		test_num = math.floor(len(files) * networkConfig.prepare_test_size)
# 		test_ids = files[0:test_num]
# 		train_ids = files[test_num:]
# 		# cut training ids if larger than limit
# 		if len(train_ids) > class_balance:
# 			train_ids = train_ids[0:class_balance]
# 		# create test class dir
# 		output_class_dir = os.path.join(test_dir, cl)
# 		if not os.path.exists(output_class_dir):
# 			os.makedirs(output_class_dir)
# 		# copy test files
# 		for image_id in test_ids:
# 			copyfile(os.path.join(source_dir, image_id), os.path.join(output_class_dir, image_id))
# 		# create train class dir
# 		output_class_dir = os.path.join(train_dir, cl)
# 		if not os.path.exists(output_class_dir):
# 			os.makedirs(output_class_dir)
# 		# copy train files
# 		for image_id in train_ids:
# 			copyfile(os.path.join(source_dir, image_id), os.path.join(output_class_dir, image_id))
# 	for i, cl in enumerate(classes):
# 		dir = os.path.join(train_dir, cl)
# 		files = [f for f in os.listdir(dir) if f.endswith(networkConfig.extend)]
# 		data_resizing(dir, files)
# 		dir = os.path.join(test_dir, cl)
# 		files = [f for f in os.listdir(dir) if f.endswith(networkConfig.extend)]
# 		data_resizing(dir, files)
# 	oprint("Dataset was successfully prepared.")
# 	return 0

# data resizing/reshaping
# def data_resizing(dir, files) :

#     # path to train directory
#     train_directory = os.path.join(networkConfig.data_location, "train")
# 	# check train directory
#     name_num = 0
#     if not os.path.exists(train_directory):
#         eprint("Train directory  not found: ", train_directory)
#     for file in files:
# 	    # stopping condition
# 	    image1 = skimage.io.imread(os.path.join(dir, file))
# 	    image = augment_img(image1, AUGMENTATIONS_TRAIN,rotation_limit=45, rotation_prob=0.5)
# 	    image = preprocess_img(image, size=image_size)
# 	    image=normalize_image(image)
# 	    new_filename = file.replace(networkConfig.extend, "")+'_{0:04d}'.format(name_num)+networkConfig.extend
# 	    os.remove(dir + '/' + file)
# 	    skimage.io.imsave(os.path.join(dir, new_filename), image)
# 	    name_num = name_num + 1;


#####################################################################
def create_ziped_dataset1(train_dataset,validation_dataset):
    classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    val_batches = tf.data.experimental.cardinality(train_dataset)
    #print(val_batches)
    classes_n = len(classes)
    train_images = []
    train_label=[]
    #trans = transforms.ToPILImage()
    print("Number of filtered classes: ", classes_n)
    class_names = train_dataset.class_names
    for images, labels in train_dataset.take(-1):
        for i in range(images.shape[0]):
            #print("  processing {}/{}, class: {}".format(i+1, classes_n, labels[i]))
            train_images.append((images[i].numpy().astype("uint8")))
            train_label.append(class_names[labels[i]])
    np.savez("./train_dataset.npz",DataX=train_images,DataY=train_label)
    #for testing
    test_images = []
    test_label=[]
    print("Number of filtered classes: ", classes_n)
    class_names = validation_dataset.class_names
    for images, labels in validation_dataset.take(-1):
        for i in range(images.shape[0]):
            #print("  processing {}/{}, class: {}".format(i+1, classes_n, labels[i]))
            test_images.append((images[i].numpy().astype("uint8")))
            test_label.append(class_names[labels[i]])
    np.savez("./test_dataset.npz",DataX=test_images,DataY=test_label)
    print("Datset ziped")

def buildDataSet():

    with np.load('train_dataset.npz',allow_pickle=True) as data:
	    x_train_origin = data['DataX']
	    y_train_origin = data['DataY']
    with np.load('test_dataset.npz',allow_pickle=True) as data:
	    x_test_origin = data['DataX']
	    y_test_origin = data['DataY']
    x_train_origin = x_train_origin.reshape(x_train_origin.shape[0], image_size[1], image_size[1],3)
    x_test_origin = x_test_origin.reshape(x_test_origin.shape[0], image_size[1], image_size[1], 3)
    dataset_train = []
    dataset_test = []
    classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    bn_classes=len(classes)
    #Sorting images by classes and normalize values 0=>1
    for n in range(0,bn_classes):
        images_class_n = np.asarray([row for idx,row in enumerate(x_train_origin) if y_train_origin[idx]==classes[n] ])
        dataset_train.append(images_class_n/255)
        images_class_test = np.asarray([row for idx,row in enumerate(x_test_origin) if y_test_origin[idx]==classes[n]  ])
        dataset_test.append(images_class_test/255)

    return dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin

######################################################################################################################

# Preparing batch for training
def buildDataSet1():
    # prepare dataset  for train and test
    PATH="."
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'test')
    BATCH_SIZE = 2800
    IMG_SIZE = (224, 224)
    train_dataset = image_dataset_from_directory(train_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    validation_dataset = image_dataset_from_directory(validation_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    for images, labels in train_dataset.take(-1):
        x_train_origin=images
        y_train_origin=labels
    # test data
    for images, labels in validation_dataset.take(-1):
        x_test_origin=images
        y_test_origin=labels
    dataset_train = []
    dataset_test = []
    #class_names = train_dataset.class_name
    #class_names1 = validation_dataset.class_names
    #Sorting images by classes and normalize values 0=>1
    n_classes=21
    classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    for n in range(0,n_classes):
        images_class_n = np.asarray([row for idx,row in enumerate(x_train_origin) if class_names[y_train_origin[idx]]==classes[n] ])
        dataset_train.append(images_class_n/255)
        images_class_n = np.asarray([row for idx,row in enumerate(x_test_origin) if class_names1[y_test_origin[idx]]==classes[n]  ])
        dataset_test.append(images_class_n/255)
    return dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin



'''
load_classes: picks classes to be loaded (influenced by MAX_CLASSES, CLASS_MIN_SAMPLES)
@params:
  source_dir - directory containing folders with individual classes
  extension - image file extension (e.g. '.png')
  min_samples - select only classes containing at least this number of classes
@returns list of filtered class names
'''
def load_classes(source_dir, extension, min_samples):
	directories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
	classes = set()
	class_sample_count = {}
	classes_count = 0
	for directory in directories:
		label = os.path.join(source_dir,directory)
		files = [os.path.join(label, f) for f in os.listdir(label) if f.endswith(extension)]
		if len(files) >= min_samples:
			classes.add(directory)
			classes_count += 1
			class_sample_count[directory] = len(files)
	classes = list(classes)
	# sort classes by name
	classes.sort()
	# print dataset info
	if False:
		print("Number of classes: ", len(classes))
		print("type: ", type(sorted(class_sample_count.items(), key=lambda item: item[1], reverse=True)))
		sorted_list = sorted(class_sample_count.items(), key=lambda item: item[1], reverse=True)
		half = math.ceil(len(sorted_list) / 2.0)
		print("sorted_list: ", sorted_list)
		for i in range(half):
			item_1 = sorted_list[i]
			item_2 = ("", 0) if i+half >= len(sorted_list) else sorted_list[i+half]
			print("%s & %s & %s & %s\\\\" % (item_1[0].replace("_", " "), item_1[1], item_2[0].replace("_", " "), item_2[1]))
			print("\\hline")
	return classes



'''
load_data: loads images and labels from directory of defined class
@params:
  source_dir - directory containing individual classes
  classes - list of class names to be selected
  extension - image file extension (e.g. '.png')
@returns - list of image_ids, list of labels
'''
def load_data(source_dir, classes, extension):
	images = []
	labels = []

	for directory in classes:
		label = os.path.join(source_dir,directory)
		files = [os.path.join(directory, f) for f in os.listdir(label) if f.endswith(extension)]

		for file in files:
			images.append(file)
			labels.append(classes.index(directory))

	return images, labels

# Read Dataset. Split into training and test set
def get_train_test_dataset(train_test_ratio):
    data_src = './train'
    X = []
    y = []
    for directory in os.listdir(data_src):
        try:
            for pic in os.listdir(os.path.join(data_src, directory)):
                img = cv2.imread(os.path.join(data_src, directory, pic))
                X.append(np.squeeze(np.asarray(img)))
                y.append(directory)
        except:
            pass

    labels = list(set(y))
    label_dict = dict(zip(labels, range(len(labels))))
    Y = np.asarray([label_dict[label] for label in y])
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    y_shuffled = []
    for index in shuffle_indices:
        x_shuffled.append(X[index])
        y_shuffled.append(Y[index])

    size_of_dataset = len(x_shuffled)
    n_train = int(np.ceil(size_of_dataset * train_test_ratio))
    n_test = int(np.ceil(size_of_dataset * (1 - train_test_ratio)))
    return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(y_shuffled[0:n_train]), np.asarray(y_shuffled[
                                                                                                  n_train + 1:size_of_dataset])