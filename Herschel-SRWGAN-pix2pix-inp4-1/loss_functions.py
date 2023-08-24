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

@tf.function()
def tfCircAperture(positions, r, xdim, ydim, x, y):
    x_pos = tf.cast(positions[:, 0], tf.float16)
    y_pos = tf.cast(positions[:, 1], tf.float16)

    aperture_tensor = tf.zeros((xdim, ydim), dtype=tf.float16)
    peak_tensor = tf.zeros((xdim, ydim), dtype=tf.float16)
    for i in tf.range(tf.shape(x_pos)[0]):
        x_diff = x - x_pos[i]
        y_diff = y - y_pos[i]
        distances = tf.sqrt(tf.square(x_diff) + tf.square(y_diff))
        aperture_tensor += tf.where(distances <= r, tf.cast(1.0, tf.float16), tf.cast(0.0, tf.float16))
        #peak_tensor += tf.where(distances <= 0.5, tf.cast(1.0, tf.float16), tf.cast(0.0, tf.float16))
        aperture_tensor.set_shape((xdim, ydim))
        peak_tensor.set_shape((xdim, ydim))
    return tf.where(aperture_tensor > 0, 1.0, 0.0), tf.where(peak_tensor > 0, 1.0, 0.0)

@tf.function
def find_local_peaks(image, threshold, box_size, center_int):
    """
    Finds local peaks in an image that are above a specified threshold value.
    Peaks are the maxima above the threshold within a local region.
    The local regions are defined by the box_size parameter.
    
    Parameters:
    image (tensorflow.Tensor): the input image
    threshold (float): the threshold value above which peaks are detected
    box_size (int): the size of the local region around each pixel
    
    Returns:
    x (tensorflow.Tensor): the x coordinates of the peaks
    y (tensorflow.Tensor): the y coordinates of the peaks
    """
    image = tf.expand_dims(image, axis=-1)
    image = tf.expand_dims(image, axis=0)
    # define a filter to find local maxima
    max_filter = tf.keras.layers.MaxPool2D(pool_size=(box_size, box_size), strides=1, padding='same')
    # apply the filter to the image
    max_image = max_filter(image)
    # find the pixels that are above the threshold and are equal to the local maxima
    mask = tf.logical_and(tf.equal(image, max_image), tf.greater(image, threshold))
    # find the indices of the peaks
    indices = tf.where(mask)
    # concatenate the x and y coordinates into a single tensor
    peaks = tf.cast(tf.stack([indices[:, 2], indices[:, 1]], axis=1), tf.int32)
    valid_mask = tf.logical_and(
        tf.logical_and(peaks[:, 0] - center_int >= box_size, peaks[:, 0] + center_int + 1 < 424 - box_size),
        tf.logical_and(peaks[:, 1] - center_int >= box_size, peaks[:, 1] + center_int + 1 < 424 - box_size)
    )
    peaks = tf.boolean_mask(peaks, valid_mask, axis=0)

    y_center = tf.cast(peaks[:, 0], tf.float32)
    x_center = tf.cast(peaks[:, 1], tf.float32)
    height = tf.fill((tf.shape(y_center)[0],), 8)
    width = tf.fill((tf.shape(x_center)[0],), 8)

    y1 = y_center - tf.cast(height / 2, tf.float32)
    x1 = x_center - tf.cast(width / 2, tf.float32)
    y2 = y_center + tf.cast(height / 2, tf.float32)
    x2 = x_center + tf.cast(width / 2, tf.float32)

    boxes = tf.stack([y1, x1, y2, x2], axis=1)

    selected_peaks = tf.image.non_max_suppression(
        tf.cast(boxes, tf.float32),
        tf.gather_nd(tf.squeeze(image), peaks),
        max_output_size=100,
        iou_threshold=0.0, # No overlap allowed
        score_threshold=threshold
    )
    return tf.gather(peaks, selected_peaks)

@tf.function
def comp_aperflux2(gen_output, target, Y_source_cat, mask):
    Laperflux = tf.constant(0, dtype=tf.float32)
    Lpeakflux = tf.constant(0, dtype=tf.float32)
    gen_output = tf.squeeze(gen_output)
    target = tf.squeeze(target)
    xdim = gen_output[0].shape[0]
    ydim = gen_output[0].shape[1]

    y, x = tf.meshgrid(tf.range(ydim), tf.range(xdim), indexing='ij')
    y = tf.cast(y, tf.float16)
    x = tf.cast(x, tf.float16)

    masked_indices = tf.squeeze(tf.where(tf.equal(mask, 1)))
    for i in tf.range(tf.shape(target)[0]):
        img_cat_mask = tf.where(tf.cast(Y_source_cat[:,-1], tf.int16) == tf.cast(masked_indices[i], tf.int16))
        img_cat_mask = tf.squeeze(img_cat_mask)
        positions = tf.gather(Y_source_cat[:,1:-1], img_cat_mask)

        gen_sources = find_local_peaks(gen_output[i], 3*2.8/1000, 40)
        if tf.shape(gen_sources)[0] > 0:
            gen_aperture_tensor, gen_peak_tensor = tfCircAperture(gen_sources, 8, xdim, ydim, x, y)
            gen_aperture_mult_gen = tf.reduce_sum(gen_aperture_tensor * gen_output[i])
            gen_aperture_mult_target = tf.reduce_sum(gen_aperture_tensor * target[i])

            Lpeakflux += tf.abs(tf.reduce_sum(gen_peak_tensor * gen_output[i]) - tf.reduce_sum(gen_peak_tensor * target[i]))/tf.cast(tf.shape(gen_sources)[0], tf.float32)
            Laperflux += tf.abs(gen_aperture_mult_gen - gen_aperture_mult_target)/tf.cast(tf.shape(gen_sources)[0], tf.float32)

        aperture_tensor, peak_tensor = tfCircAperture(positions, 8, xdim, ydim, x, y)

        aperture_mult_gen = tf.reduce_sum(aperture_tensor * gen_output[i])
        aperture_mult_target = tf.reduce_sum(aperture_tensor * target[i])
        Lpeakflux += tf.abs(tf.reduce_sum(peak_tensor * gen_output[i]) - tf.cast(tf.reduce_sum(tf.gather(Y_source_cat[:,0], img_cat_mask)), tf.float32))/tf.cast(tf.shape(img_cat_mask)[0], tf.float32)#tf.cast(tf.reduce_sum(tf.gather(Y_source_cat[:,0], mask)), tf.float32))/tf.cast(tf.shape(mask)[0], tf.float32)
        Laperflux += tf.abs(aperture_mult_gen - aperture_mult_target)/tf.cast(tf.shape(img_cat_mask)[0], tf.float32)
    return Laperflux, Lpeakflux

@tf.function
def gaussian2D(x, y, x0, y0, sigma_x, sigma_y):
    """
    Computes the value of a 2D Gaussian function at (x, y).
    """
    return tf.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

import tensorflow as tf


# @tf.function
def kernel_loss(gen_output, Y, true_kernel):
    # FWHM = tf.constant(7.9, dtype=tf.float32)
    # sigma = FWHM/tf.cast(2.355, tf.float32)
    loss = tf.constant(0., dtype=tf.float32)
    kernel_size = true_kernel.shape[0]
    # # Define the x and y coordinates.
    # xx, yy = tf.meshgrid(tf.range(0, kernel_size, 1), tf.range(0, kernel_size, 1))
    # xx = tf.cast(xx, tf.float32)
    # yy = tf.cast(yy, tf.float32)
    # Hardcoded noise level
    std_noise = 2./1000 #mJy
    # Find the locations of sources whose peaks are above the given threshold
    # The minimum separation distance is given by the FWHM
    center_int = tf.cast(tf.round((kernel_size - 1)/2), tf.int32)
    # center_float = tf.cast(tf.round((kernel_size - 1)/2), tf.float32)

    # kernel = gaussian2D(xx, yy, center_float, center_float, sigma, sigma)
    # # plt.figure(figsize=(10, 10))
    # # plt.imshow(kernel)
    # # plt.show()
   
    gen_output = tf.squeeze(gen_output)
    # Loop over image batch
    for i in tf.range(Y.shape[0]):
        kernel_gen = tf.zeros((kernel_size, kernel_size), dtype=tf.float32)
        peaks = find_local_peaks(gen_output[i], 4.5*std_noise, 8, center_int)
        peaks = tf.cast(tf.round(peaks), tf.int32)
        
        # Only process non-empty peaks
        if peaks.shape[0] != 0:
            # Create a mask for the valid peaks
            # valid_mask = tf.logical_and(
            #     tf.logical_and(peaks[:, 0] - center_int >= 0, peaks[:, 0] + center_int + 1 < 424),
            #     tf.logical_and(peaks[:, 1] - center_int >= 0, peaks[:, 1] + center_int + 1 < 424)
            # )
            # valid_peaks = tf.boolean_mask(peaks, valid_mask, axis=0)
            
            if peaks.shape[0] == 0: continue
            # Add valid peaks to kernel
            for j in tf.range(peaks.shape[0]):
                peak_x = tf.round(peaks[j][0])
                peak_y = tf.round(peaks[j][1])

                kernel_gen += gen_output[i, peak_y - center_int:peak_y + center_int + 1, peak_x - center_int:peak_x + center_int + 1]
        

            kernel_gen /= tf.reduce_max(kernel_gen)

            loss += tf.reduce_mean(tf.abs(kernel_gen - true_kernel))

    return loss/tf.cast(Y.shape[0], tf.float32)

# Function used during weight training
@tf.function()
def non_adversarial_loss(gen_output, target):
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    Lh = huber_loss(gen_output, target)
    # Lk = tf.constant(0., dtype=tf.float32)
    #Lflux = tf.numpy_function(comp_aperflux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    #Lflux = comp_aperflux(gen_output, target, Y_source_cat)#tf.py_function(comp_peakflux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    #Laperflux, Lpeakflux = comp_aperflux(gen_output, target, Y_source_cat, mask)
    # Ldisc = cross_entropy(tf.ones_like(fake_output), fake_output)
    return Lh

# Function used for validation loss computations
def non_adversarial_loss_valid(gen_output, target, true_kernel):
    Lh = huber_loss(gen_output, target)
    #Laperflux, Lpeakflux = comp_aperflux(gen_output, target, Y_source_cat, mask)
    Lk = kernel_loss(gen_output, target, true_kernel)
    # Lk = tf.constant(0., dtype=tf.float32)
    #Lflux = comp_aperflux(gen_output, target, Y_source_cat)
    #Lfluxpeak = comp_peakflux(gen_output, target, Y_source_cat)
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Ldisc = cross_entropy(tf.ones_like(fake_output), fake_output)
    return Lh, Lk

@tf.function
def Wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(y_true * y_pred)
    
def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return np.mean(np.equal(y_true, np.array(y_pred>= 0.5).astype(int)))
    