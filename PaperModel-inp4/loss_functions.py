######################
###     IMPORTS    ###
######################


import time
import math as m
import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

from matplotlib import pyplot as plt
from astropy import stats

from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources
from photutils.aperture import CircularAperture, aperture_photometry

from multiprocessing import Pool

######################
### LOSS FUNCTIONS ###
######################
# @tf.function
# def huber_loss(gen_output, target, δ):
#     err = gen_output - target
#     err_abs = tf.abs(err)
#     Lh = tf.where(err_abs < δ, 0.5 * tf.square(err), δ * (err_abs - 0.5 * δ))
#     return tf.reduce_sum(Lh)

@tf.function
def huber_loss(gen_output, target):
    x = gen_output - target
    return tf.reduce_sum(x + tf.math.softplus(-2.0 * x) - tf.cast(tf.math.log(2.0), x.dtype))/tf.cast(tf.shape(target)[0], tf.float32)

@tf.function
def L1(gen_output, target):
    return 100*tf.reduce_mean(tf.abs(gen_output - target))

# Function used during weight training
@tf.function()
def non_adversarial_loss(gen_output, target):
    L = L1(gen_output, target)
    return L

# Function used for validation loss computations
def non_adversarial_loss_valid(gen_output, target):
    L = huber_loss(gen_output, target)
    return L

@tf.function
def Wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(y_true * y_pred)
    
def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return np.mean(np.equal(y_true, np.array(y_pred>= 0.5).astype(int)))
    