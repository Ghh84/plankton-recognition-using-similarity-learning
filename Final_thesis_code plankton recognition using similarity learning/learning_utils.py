from data_preprocess_functions import load_classes
import matplotlib.pyplot as plt
import numpy as np
#from keras.engine.topology import Layer
from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import math
from sklearn.metrics import roc_curve, auc, roc_auc_score

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

classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
nb_classes=len(classes)
input_shape = (224,224, 3)
image_size = (224, 224 )

def get_batch_random(batch_size,dataset_train,s="train"):
    """
    Create batch of APN triplets with a complete random strategy
    Arguments:
    batch_size -- integer
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    if s == 'train':
        X = dataset_train
    else:
        X = dataset_test
    m, w, h,c = X[0].shape
    classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    nb_classes=len(classes)
    
    # initialize result
    triplets=[np.zeros((batch_size,h, w,c)) for i in range(3)]
    for i in range(batch_size):
	    #Pick one random class for anchor
	    anchor_class = np.random.randint(0, nb_classes)
	    nb_sample_available_for_class_AP = X[anchor_class].shape[0]
	    #Pick two different random pics for this class => A and P
	    [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)
	    #Pick another class for N, different from anchor_class
	    negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
	    nb_sample_available_for_class_N = X[negative_class].shape[0]
	    #Pick a random pic for this negative class => N
	    idx_N = np.random.randint(0, nb_sample_available_for_class_N)
	    triplets[0][i,:,:,:] = X[anchor_class][idx_A,:,:,:]
	    triplets[1][i,:,:,:] = X[anchor_class][idx_P,:,:,:]
	    triplets[2][i,:,:,:] = X[negative_class][idx_N,:,:,:]
    return triplets

def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,dataset_train,s="train"):
    """
    Create batch of APN "hard" triplets
    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    if s == 'train':
        X = dataset_train
    else:
        X = dataset_test
    m, w, h,c = X[0].shape
    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,dataset_train,s)
    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
    #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])
    #Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
    #Sort by distance (high distance first) and take the
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)
    selection = np.append(selection,selection2)
    triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:], studybatch[2][selection,:,:,:]]
    return triplets
'''
Compute the distance between two extracted features
'''
def compute_dist(a,b):
    return np.sum(np.square(a-b))
'''
Compute the euclidean distance between two extracted features
'''
def euclidean_distance(x,y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#Validation / evaluation
def compute_probs(network,X,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class
    Returns
        probs : array of shape (m,m) containing distances

    '''
    m = X.shape[0]
    nbevaluation = int(m*(m-1)/2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))
    #Compute all embeddings for all pics with current network
    embeddings = network.predict(X)
    size_embedding = embeddings.shape[1]
    #For each pics of our dataset
    k = 0
    for i in range(m):
            #Against all other images
            for j in range(i+1,m):
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])
                if (Y[i]==Y[j]):
                    y[k] = 1
                    #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
                else:
                    y[k] = 0
                    #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
                k += 1
    return probs,y

def compute_metrics(probs,yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)
    return fpr, tpr, thresholds,auc

def compute_interdist(network,dataset_test):
    '''
    Computes sum of distances between all classes embeddings on our reference test image:
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings

    Returns:
        array of shape (nb_classes,nb_classes)
    '''
    #classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    #nb_classes=len(classes)
    res = np.zeros((nb_classes,nb_classes,3))
    ref_images = np.zeros((nb_classes,image_size[1],image_size[1],3))
    print(ref_images.shape)
    #generates embeddings for reference images
    for i in range(nb_classes):
        ref_images[i,:,:,] = dataset_test[i][0,:,:,:]
    ref_embeddings = network.predict(ref_images)

    for i in range(nb_classes):
        for j in range(nb_classes):
            res[i,j] = compute_dist(ref_embeddings[i],ref_embeddings[j])
    return res

def draw_interdist(network,n_iteration,dataset_test):
    interdist = compute_interdist(network,dataset_test)
    data = []
    for i in range(nb_classes):
        data.append(np.delete(interdist[i,:],[i]))
    fig, ax = plt.subplots()
    ax.set_title('Evaluating embeddings distance from each other after {0} iterations'.format(n_iteration))
    ax.set_ylim([0, 3])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data,showfliers=False,showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs,np.arange(nb_classes))
    plt.show()

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1],idx-1
    else:
        return array[idx],idx
    
def draw_roc(fpr, tpr,thresholds,auc):
    #find threshold
    targetfpr=1e-2
    _, idx = find_nearest(fpr,targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,targetfpr,recall,abs(threshold) ))
    # show the plot
    plt.show()  
'''
   draw or plot sample test or train image with distance between them

'''
def draw_test_images(network, images, dataset_test,indx, threshold, classindicator=-1, refidx=0,nb_test_class=50 ):
    predicted=[]
    _, w,h,c = dataset_test[0].shape
    nbimages=images.shape[0]
    #generates embedings for given images
    image_embedings = network.predict(images)
    # planton spieces/ classes 
    classes = load_classes(networkConfig.data_location, networkConfig.extend, networkConfig.min_samples)
    nb_display=len(classes)
    nb_test_class=len(classes)
    #generates embedings for reference images
    ref_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        ref_images[i,:,:,:] = dataset_test[i][refidx,:,:,:]
    ref_embedings = network.predict(ref_images)
    for i in range(nbimages):
        if nbimages>1:
            trueclass=i
        else:
            trueclass=classindicator
        
        #Prepare the figure
        fig=plt.figure(figsize=(50,50))
        subplot = fig.add_subplot(1,nb_display+1,1)
        subplot.axis("off")
        plotidx = 2
            
        #Draw this image    
        plt.imshow(images[i,:,:,0])
        subplot.title.set_text("Test image")
            
        distdtype=[('class', int), ('dist', float)]
        dist = np.zeros(nb_test_class, dtype=distdtype)
        
        #Compute distances
        for ref in range(nb_test_class):
            #Compute distance between this images and references
            dist[ref] = (ref,compute_dist(image_embedings[i,:],ref_embedings[ref,:]))
        #sort
        sorted_dist = np.sort(dist, order='dist')
        predicted.append(classes[sorted_dist['class'][0]])
        #Draw
        for j in range(min(20,nb_test_class)):
            subplot = fig.add_subplot(1,nb_display+1,plotidx)
            plt.imshow(ref_images[sorted_dist['class'][j],:,:,0])
            subplot.axis("off")
            #Red for sample above threshold
            if (sorted_dist['dist'][j] > threshold):
                if (trueclass == sorted_dist['class'][j]):
                    color = (1,0,0)
                    label = "TRUE"
                else:
                    color = (0.5,0,0)
                    label = "Class {0}".format(sorted_dist['class'][j])
            else:
                if (trueclass == sorted_dist['class'][j]):
                    color = (0, 1, 0)
                    label = "TRUE"
                else:
                    color = (0, .5, 0)
                    label = "Class {0}".format(sorted_dist['class'][j])
                
            subplot.set_title("{0}\n{1:.3e}".format(label,sorted_dist['dist'][j]),color=color)
            plotidx += 1
    return predicted
