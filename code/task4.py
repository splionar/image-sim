import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

# Feel free to add as many imports as you need
from random import randint


def crop_image(img, random=False):
    """
    Takes an image as input and returns a copy of it with a 
    missing region, as well as the corresponding missing region

    INPUT:
    - img: a numpy array of size (heigh, width, 3) 
    - random: if true, the crop is taken at a random position,
    if false, the crop is taken at the center
    /!\ IGNORE FOR TASK 1

    OUTPUT:
    - img_with_a_hole: a numpy array of size (heigh, width, 3)
    - missing_region: a numpy array of size (64, 64, 3)

    HINT:
    For task 6, change the default random value to True
    """    
    h, w, _ = np.shape(img)
    
    img1 = img[:,:int(w/3)].copy()
    img23 = img[:,int(w/3):].copy()

        
    return img1, img23


def create_reconstruction_model():
    """
    Create a keras sequential model that reproduces figure 9.a
    of the paper

    OUTPUT:
    - model: a keras sequential model
    """
    model = Sequential()

    # Encoder
    model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding = 'same', input_shape=(128,128,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Bottleneck
    model.add(Conv2D(4000, kernel_size=4, strides=(1,1), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    # Decoder
    model.add(Conv2DTranspose(512, kernel_size=4, strides=(2, 2), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=(2, 2), padding = 'same'))

    return model


def reconstruction_loss(predicted_region, groundtruth):
    """
    Computes the loss between the predicted region and the 
    corresponding groundtruth.

    INPUT: 
    - predicted_region: a tensor of shape (batch_size, 64, 64, 3)
    - groundtruth: a tensor of shape (batch_size, 64, 64, 3)

    OUTPUT:
    - loss_value: a tensor scalar

    HINT:
    Functions that might be useful (but you can use any tensorflow function you find
    useful, not necessarily those):
    - tf.reduce_mean
    - tf.square
    - tf.reduce_sum
    """
    error_similar = predicted_region - groundtruth[:, :, :128]
    error_dissimilar = predicted_region - groundtruth[:, :,128:]

   
    loss_similar = tf.reduce_mean(tf.square(error_similar))
    loss_dissimilar =  tf.reduce_mean(tf.square(error_dissimilar))
    
    loss = 0.6 * loss_similar - 0.4 * loss_dissimilar
    
    return loss


def reconstruct_input_image(input_data, predicted_region):
    """
    Combines an input image (with a hole), and a (predicted) missing region
    to produce a full image.

    INPUT:
    - input_data: a numpy array of size (height, width, 3)
    - predicted_region: a numpy array of size (64, 64, 3)

    OUTPUT:
    - full_image: a numpy array of size (height, width, 3)
    """
    

    return predicted_region
