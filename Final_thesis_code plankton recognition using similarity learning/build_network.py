#from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Dropout,BatchNormalization,Dense,Lambda, Flatten
from tensorflow.keras.models import Model
#from tensorflow.keras.layers.pooling import MaxPooling2D
#from tensorflow.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.keras.regularizers import l2
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import os
import tensorflow as tf
#from tensorflow.keras.preprocessing import image_dataset_from_directory

#os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

#nb_classes=21;
image_size = (224,224)
'''
    Define the neural network to learn image similarity
    Input :
    input_shape : shape of input images 
    name: name of CNN architecture
    embeddingsize : vectorsize used to encode our picture
'''
def build_network_pretrained(input_shape, name,embeddingsize):
    
    if (name=="VGG16"):
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        import numpy as np
        base_model = VGG16(input_shape=input_shape,weights='imagenet', include_top=False)
        net =base_model.output
        net = tf.keras.layers.Dropout(0.5)(net)
        net = tf.keras.layers.Flatten(name='new_flatten')(net) 
        net = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-3), 
                                    kernel_initializer='he_uniform',name='new_fc')(net)
        net = tf.keras.layers.Dense(embeddingsize, activation=None, kernel_regularizer=l2(1e-3), 
                                    kernel_initializer='he_uniform', name='new_fc2')(net)
        #Force the encoding to live on the d-dimentional hypershpere
        net = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1),name='lamda')(net)
        base_model =tf.keras.Model(base_model.input, net, name='tuned_model')
        #base_model.trainable = False
        # Let's take a look to see how many layers are in the base model
        # Fine-tune from this layer onwards
        fine_tune_at = 18
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
                layer.trainable =  False
        #base_model.get_layer('block5_conv3').trainable = True
        #base_model.get_layer('fc1').trainable = True
        #base_model.get_layer('fc2').trainable = True
        #base_model.get_layer('new_fc').trainable = True
        #base_model.get_layer('new_fc2').trainable = True
        #base_model.get_layer('lamda').trainable = True
       
        
    elif(name=="VGG19"):
        from tensorflow.keras.applications.vgg19 import VGG19
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.vgg19 import preprocess_input
        from tensorflow.keras.models import Model
        base_model = VGG19(input_shape=input_shape,weights='imagenet')
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        embeddingsize=50
        net =base_model.output
        net = tf.keras.layers.Dropout(0.5)(net)
        net = tf.keras.layers.Flatten()(net) 
        net = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(net)
        net = tf.keras.layers.Dense(embeddingsize, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(net)
        #Force the encoding to live on the d-dimentional hypershpere
        net = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))(net)
        base_model =tf.keras.Model(base_model.input, net, name='tuned_model')
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        # Fine-tune from this layer onwards
        fine_tune_at = 18
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
                layer.trainable =  False
       
        
    elif(name=="mobileNet"):
        # Create the base model from the pre-trained model MobileNet V2
        from tensorflow.keras.applications import MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        embeddingsize=50
        net =base_model.output
        net = tf.keras.layers.Dropout(0.3)(net)
        net = tf.keras.layers.Flatten()(net) 
        #net = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(net)
        net = tf.keras.layers.Dense(embeddingsize, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(net)
        #Force the encoding to live on the d-dimentional hypershpere
        net = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))(net)
        base_model =tf.keras.Model(base_model.input, net, name='tuned_model')
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        # Fine-tune from this layer onwards
        fine_tune_at = 120
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
                layer.trainable =  False

    elif(name=="resNet50"):
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
        import numpy as np
        base_model = ResNet50(input_shape=input_shape,weights='imagenet', include_top = False)
        # Add fully connected layer which have 1024 neuron to ResNet-50 model
        embeddingsize=50
        net =base_model.output
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Flatten()(net) 
        net = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-2), kernel_initializer='he_uniform')(net)
        net = tf.keras.layers.Dense(embeddingsize, activation=None, kernel_regularizer=l2(1e-2), kernel_initializer='he_uniform')(net)
        #net = tf.keras.layers.Flatten()(net) 
        #Force the encoding to live on the d-dimentional hypershpere
        net = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))(net)
        base_model =tf.keras.Model(base_model.input, net, name='tuned_model')
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
                layer.trainable =  False

    elif(name=="inceptionV3"):
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D   
        # create the base pre-trained model
        base_model = InceptionV3(input_shape=input_shape,weights='imagenet', include_top=False) 
        net =base_model.output
        net = tf.keras.layers.Dropout(0.5)(net)
        net = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(net)
        net = tf.keras.layers.Dense(embeddingsize, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform')(net)
        net = tf.keras.layers.Flatten()(net) 
        #Force the encoding to live on the d-dimentional hypershpere
        net = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))(net)
        base_model =tf.keras.Model(base_model.input, net, name='tuned_model')
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        # Fine-tune from this layer onwards
        fine_tune_at = 220
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False 
    else:
        # Convolutional Neural Network
        base_model = Sequential()
        base_model.add(Conv2D(64, (7,7), activation='relu',input_shape=input_shape, kernel_initializer='he_uniform',
                              kernel_regularizer=l2(2e-4)))
        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
                              kernel_regularizer=l2(2e-4)))
        base_model.add(MaxPooling2D())
        base_model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',
                              kernel_regularizer=l2(2e-4)))
        base_model.add(Flatten())
        base_model.add(Dropout(0.5))
        base_model.add(Dense(256, activation='relu',kernel_regularizer=l2(1e-3),kernel_initializer='he_uniform'))
        base_model.add(Dropout(0.5))
        base_model.add(Dense(embeddingsize, activation=None,kernel_regularizer=l2(1e-3),
                          kernel_initializer='he_uniform'))
        base_model.add(Dropout(0.5))
        #Force the encoding to live on the d-dimentional hypershpere
        base_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
        print('No such pretrained model')
    return base_model

def build_model(input_shape, network, margin=0.02):
    '''
    Define the Keras Model for training
        Input :
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    '''
     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")
    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    # return the model
    return network_train














# Get ResNet-50 Model
def getResNet50Model(lastFourTrainable=True):
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np
    resnet_model = ResNet50(weights='imagenet', input_shape=(224,224,3), include_top=True)
    # Make all layers non-trainable
    for layer in resnet_model.layers[:]:
        layer.trainable = False
    # Add fully connected layer which have 1024 neuron to ResNet-50 model
    output = resnet_model.get_layer('avg_pool').output
    output = tf.keras.layers.Flatten(name='new_flatten')(output)
    output = tf.keras.layers.Dense(1024, activation='relu', name='new_fc')(output)
    output = tf.keras.layers.Dense(50, activation=None,  name='new_fc1')(output)
    #output = tf.keras.layers.Dense(50, activation=None)(output)
    #Force the encoding to live on the d-dimentional hypershpere
    output = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1), name='lamda')(output)
    resnet_model = tf.keras.Model(resnet_model.input, output)
    # Make last 4 layers trainable if lastFourTrainable == True
    resnet_model.get_layer('conv5_block3_2_bn').trainable = True
    resnet_model.get_layer('conv5_block3_3_conv').trainable = True
    resnet_model.get_layer('conv5_block3_3_bn').trainable = True
    resnet_model.get_layer('new_fc').trainable = True
    resnet_model.get_layer('new_fc1').trainable = True
    resnet_model.get_layer('lamda').trainable = True
    return resnet_model 

