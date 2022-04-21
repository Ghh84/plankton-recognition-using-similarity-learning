import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from data_preprocess_functions import create_ziped_dataset1,buildDataSet,load_classes
from build_network import build_model,build_network_pretrained,getResNet50Model
from utils import DrawPics,drawTriplets
from learning_utils import compute_probs,get_batch_hard,get_batch_random,euclidean_distance,compute_dist,compute_metrics,compute_interdist,draw_interdist,draw_roc
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve,roc_auc_score
#from keras import backend as K
from tensorflow.keras.optimizers import Adam
import time
import os
import os.path
from os import path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.regularizers import l2
#from keras import backend as K
from tensorflow.keras.utils import plot_model,normalize

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
class NetworkConfig:
	def __init__(self, min_samples=0):
		self.min_samples = min_samples

networkConfig = NetworkConfig()
networkConfig.model_path = None
networkConfig.min_samples = 1000
networkConfig.data_location = "./labeled_20201020"
networkConfig.extend = '.png'
networkConfig.prepare_dir = ''
networkConfig.prepare_num = 10
networkConfig.prepare_test_size = 20/ float(100)
output_dir="."

'''
Default parameters
'''
projectName = "Weights"
project_path = './{0}/'.format(projectName)
model_path = '../{0}/'.format(projectName)
if not path.exists(project_path):
    os.mkdir(project_path)
if not path.exists(model_path):
    os.mkdir(model_path)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
input_shape=IMG_SHAPE
image_size = (224, 224 )
nb_classes=50
''' 
set training parameters 
'''
evaluate_every = 100 # interval for evaluating on one-shot tasks
batch_size = 16
n_iter = 10000# No. of training iterations
n_val = 100 # how many one- shot tasks to validate on
## # #
t_start = time.time()
n_iteration=0
loss_history=[];
file = open("store.txt", "w")

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)
def triplet_loss(y_true, y_pred):
    margin = K.constant(0.02)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))
def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
# Trainning of the model


def prediction1(network, x_test_origin, dataset_test, test_label,top,threshold,classindicator=-1, refidx=0,nb_test_class=50 ):
    predicted=[]
    _, w,h,c = dataset_test[0].shape
    nbimages=x_test_origin.shape[0]
    #nb_display=50
    # planton spieces/ classes
    classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    nb_display=len(classes)
    nb_test_class=len(classes)
    #generates embedings for reference images
    ref_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        ref_images[i,:,:,:] = dataset_test[i][refidx,:,:,:]
    ref_embedings = network.predict(ref_images)
    c=0
    for i in range(0,nbimages):
        #generates embedings for given images
        image_embedings = network.predict(np.expand_dims(x_test_origin[i,:,:,:],axis=0))
        if nbimages>1:
            trueclass=i
        else:
            trueclass=classindicator
        distdtype=[('class', int), ('dist', float)]
        dist = np.zeros(nb_test_class, dtype=distdtype)
        #Compute distances
        for ref in range(nb_test_class):
            #Compute distance between this images and references
            dist[ref] = (ref,compute_dist(image_embedings[:],ref_embedings[ref,:]))
        #sort
        sorted_dist = np.sort(dist, order='dist')
        predicted.append(classes[sorted_dist['class'][0]])
    return predicted

'''
Prepare dataset 
'''

#prepare_dataset(networkConfig, output_dir, class_balance=10, augment=True)
[dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin]=buildDataSet()
print("Checking shapes for class 0 (train) : ",dataset_train[0].shape)
print("Checking shapes for class 0 (test) : ",dataset_test[0].shape)

base_model=build_network_pretrained(input_shape,"VGG16",50)
#base_model=build_network_pretrained(input_shape,"mobileNet",50)
#base_model=getResNet50Model()
model = build_model(input_shape,base_model)
optimizer = Adam(lr = 0.00006)
model.compile(loss=None,optimizer=optimizer)
#model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer="rmsprop", loss=triplet_loss, metrics=[accuracy])
model.summary()
#model.load_weights('{0}Best1_VGG19_50.h5'.format(model_path))
#model.load_weights('{0}mobileNet_new_1.h5'.format(model_path))
#model.load_weights('{0}resnet_new_2.h5'.format(model_path))
#model.load_weights('{0}resnet_new_1.h5'.format(model_path))
model.load_weights('{0}Best2_VGG16_14.h5'.format(model_path))
#plot_model(network_train,show_shapes=True, show_layer_names=True, to_file='02 model.png')
#print(model.metrics_names)
#n_iteration=0
#network_train.load_weights('mnist-160k_weights.h5')
#top=1
#predicted=prediction1(base_model, x_test_origin, dataset_train,y_test_origin,top, threshold=abs(.3))
#Top1_count=np.sum(predicted == y_test_origin)
#print(Top1_count)

'''
Train the model 
'''

for i in range(1, n_iter+1):
    #triplets = get_batch_random(batch_size,dataset_train)
    triplets=get_batch_hard(200,16,16,base_model,dataset_train)
    loss = model.train_on_batch(triplets, None)
    loss_history.append(loss)
    #network_train.save_weights(file)
    n_iteration += 1
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))
        file.write("{}\n" .format(loss))
    if i % 5000 ==0:
        top=1
        predicted=prediction1(base_model, x_test_origin, dataset_train,y_test_origin,top, threshold=abs(.3))
        Top1_count=np.sum(predicted == y_test_origin)
        print(Top1_count)
#Final save
model.load_weights('{0}Best1_VGG16_50.h5'.format(model_path))
#model.save('{0}model_VGG19.h5'.format(model_path))
#model.save_weights('{0}test9_VGG19_1000.h5'.format(model_path))
top=1
predicted=prediction1(base_model, x_test_origin, dataset_test,y_test_origin,top, threshold=abs(1))
Top1_count=np.sum(predicted == y_test_origin)
print(Top1_count)






