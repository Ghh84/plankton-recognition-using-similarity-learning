from data_preprocess_functions import prepare_dataset,create_ziped_dataset,buildDataSet,load_classes
from build_network import build_network,build_network1,build_network2,build_model,build_model_alexnet,build_network_n
from utils import DrawPics,drawTriplets
from learning_utils import compute_probs,compute_dist,compute_metrics,compute_interdist,draw_interdist,draw_roc,DrawTestImage, get_batch_hard,get_batch_random
from keras.engine.topology import Layer
from sklearn.metrics import roc_curve,roc_auc_score
from keras import backend as K
from keras.optimizers import Adam
import time
import os
import os.path
from os import path
import tensorflow as tf
#############################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC']='1'

#random.seed(42)
#np.random.seed(42)
tf.random.set_seed(42)

#################################
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
networkConfig.min_samples = 200
networkConfig.data_location = "/content/drive/MyDrive/plankton_recognition/labeled"
networkConfig.extend = '.png'
networkConfig.prepare_dir = ''
networkConfig.prepare_num = 10
networkConfig.prepare_test_size = 20/ float(100)
output_dir="/content/drive/MyDrive/plankton_recognition"
input_shape = (224,224, 1)
image_size = (224, 224 )

#.......................................................................................................#
'''
prepare_dataset: Prepares dataset for training and testing
'''
prepare_dataset(networkConfig, output_dir, class_balance=40, augment=False)
#nb_classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
##[im,la]= load_data(source_dir, classes, '.png' )
##########################################################################################################
#'''
#preparation of .nzp datasets
#'''
create_ziped_dataset();
#
#'''
#building of dataset for train and test
#'''
[dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin]=buildDataSet()
print("Checking shapes for class 0 (train) : ",dataset_train[0].shape)
print("Checking shapes for class 0 (test) : ",dataset_test[0].shape)
print("Checking first samples")
#for i in range(7,12):
     #DrawPics(dataset_train[i],5,template='Train {}',classnumber=i)
     #DrawPics(dataset_test[i],5,template='Test {}',classnumber=i)

## # ############################################################################################################

batch_size=2
triplets=get_batch_random(batch_size,dataset_train)
#print("Checking batch width, should be 3 : ",len(triplets))
#print("Shapes in the batch A:{0} P:{1} N:{2}".format(triplets[0].shape, triplets[1].shape, triplets[2].shape))
#drawTriplets(triplets)
## ###########################################################################################################
#
## ### define network"
#network=build_network_n(input_shape,embeddingsize=10)
#network = build_network(input_shape,embeddingsize=nb_classes)
#network = build_network1(input_shape,embeddingsize=12)
#network = build_network2(input_shape,embeddingsize=nb_classes)
#network=build_model_alexnet(input_shape,nb_classes, image_size)
#network_train = build_model(input_shape,network)
#optimizer = Adam(lr = 0.00006)
#network_train.compile(loss=None,optimizer=optimizer)
#network_train.summary()
##
##
#hardtriplets = get_batch_hard(50,1,1,network,dataset_train)
#print("Shapes in the hardbatch A:{0} P:{1} N:{2}".format(hardtriplets[0].shape, hardtriplets[1].shape, hardtriplets[2].shape))
#drawTriplets(hardtriplets)
# ##
## ##Hyper parameters
#evaluate_every = 2 # interval for evaluating on one-shot tasks
#batch_size = 32
#n_iter = 800# No. of training iterations
#n_val = 250 # how many one- shot tasks to validate on
## # #
## ###Testing on an untrained network
#probs,yprob = compute_probs(network,x_test_origin[:500,:,:,:],y_test_origin[:500])
#fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
#draw_roc(fpr, tpr,thresholds,auc)
#draw_interdist(network,0,dataset_test)
## #
#DrawTestImage(network,dataset_test, np.expand_dims(dataset_train[1][0,:,:,:],axis=0))
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#projectName = "train_out"
#project_path = './{0}/'.format(projectName)
#model_path = '../{0}/'.format(projectName)
#if not path.exists(project_path):
    #os.mkdir(project_path)
#if not path.exists(model_path):
    #os.mkdir(model_path)
print("Starting training process!")
print("-------------------------------------")
##Allow memory growth for the GPU
t_start = time.time()
n_iteration=0
file = open("/content/drive/MyDrive/plankton_recognition/store.txt", "w")
for i in range(1, n_iter+1):
    triplets = get_batch_hard(100,16,16,network,dataset_train)
    loss = network_train.train_on_batch(triplets, None)
    n_iteration += 1
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))
        probs,yprob = compute_probs(network,x_test_origin[:n_val,:,:,:],y_test_origin[:n_val])
        #network_train.save_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
        file.write("{}\n" .format(loss))
###Final save
#network_train.save_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iter,model_path))
#fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
#draw_roc(fpr, tpr,thresholds)
#draw_interdist(network,n_iter)
#for i in range(1,len(probs)):
#	file.write(" %s," %( probs[i]))
#file.write("\n%s\n" %( "yprob"))
#for i in range(1,len(probs)):
#	file.write("%s," %( yprob[i]))
#	file.write("%s = %s\n" %("loss", loss))
file.close()
print("Done !")

#############################################################################################

#Full evaluation
probs,yprob = compute_probs(network,x_test_origin,y_test_origin)
fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
draw_roc(fpr, tpr,thresholds)
draw_interdist(network,n_iter)


















