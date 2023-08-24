import os
import gc
import time
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity as ssim
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy import stats
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources
from photutils.aperture import CircularAperture, aperture_photometry

import imageio
import glob

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Setting random seed
tf.random.set_seed(42)

# Perform garbage collection
gc.collect()


def huber_loss(gen_output, target):
    err = gen_output - target
    return tf.reduce_sum(tf.math.log(tf.math.cosh(err)))
    
def comp_stats(gen_output, target):
    def comp_mean_median(x, sigma):
        mean, median, _ = stats.sigma_clipped_stats(x, sigma=sigma, maxiters=3)
        return np.float32(mean), np.float32(median)
    gen_mean, gen_median = tf.numpy_function(comp_mean_median, [gen_output, 3], Tout=[tf.float32, tf.float32])
    target_mean, target_median = tf.numpy_function(comp_mean_median, [target, 3], Tout=[tf.float32, tf.float32])
    l_mean = tf.abs(gen_mean - target_mean)
    l_median = tf.abs(gen_median - target_median)
    return l_mean + l_median

def compute_PSNR_batch(gen_output, target):
    MSE = np.mean((gen_output - target)**2)
    return 20*np.log10(1/np.sqrt(MSE))

def comp_peakflux(gen_output, target, Y_source_cat):
    total_peakflux_diff = 0
    gen_output = np.squeeze(gen_output)
    target = np.squeeze(target)
    apertures = CircularAperture(Y_source_cat[:,1:-1], r=8)
    for i in tf.range(target.shape[0]):
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        aperture_masks = apertures[mask].to_mask(method='center')
        for idx, k in enumerate(mask):
            total_peakflux_diff += abs(Y_source_cat[k,0] - np.max(aperture_masks[idx].multiply(gen_output[i])))/len(mask)
    return np.float32(total_peakflux_diff)

def can_connect_circles(circles, PSF):
    # Depth-First Search based method
    def distance(c1, c2):
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def can_connect(c1, c2):
        return distance(c1, c2) <= PSF

    def dfs(node, visited):
        visited.add(node)
        for neighbor in adjacency_list[node]:
            if neighbor not in visited:
                if dfs(neighbor, visited):
                    return True
        return False

    # create an adjacency list representation of the graph
    adjacency_list = {i: [] for i in range(len(circles))}
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            if can_connect(circles[i], circles[j]):
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    # check if all circles can be connected using a depth-first search
    visited = set()
    for i in range(len(circles)):
        if dfs(i, visited):
            return True
    return False

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
        tf.logical_and(peaks[:, 0] - center_int >= 0, peaks[:, 0] + center_int + 1 < 424),
        tf.logical_and(peaks[:, 1] - center_int >= 0, peaks[:, 1] + center_int + 1 < 424)
    )
    peaks = tf.boolean_mask(peaks, valid_mask, axis=0)


    y_center = tf.cast(peaks[:, 0], tf.float32)
    x_center = tf.cast(peaks[:, 1], tf.float32)
    height = tf.fill((tf.shape(y_center)[0],), 8)
    width = tf.fill((tf.shape(x_center)[0],), 8)

    y1 = y_center - tf.cast(height, tf.float32)
    x1 = x_center - tf.cast(width, tf.float32)
    y2 = y_center + tf.cast(height, tf.float32)
    x2 = x_center + tf.cast(width, tf.float32)

    boxes = tf.stack([y1, x1, y2, x2], axis=1)


    selected_peaks = tf.image.non_max_suppression(
        tf.cast(boxes, tf.float32),
        tf.gather_nd(tf.squeeze(image), peaks[:, ::-1]),
        max_output_size=100,
        iou_threshold=0.0, # No overlap allowed
        score_threshold=threshold
    )
    return peaks, tf.gather(peaks, selected_peaks)


def evaluate_custom_source_detection(img, window_size, kernel_size):
    std_noise = 2./1000 #mJy
    center_int = tf.cast(tf.round((kernel_size - 1)/2), tf.int32)
    peaks, supressed_peaks = find_local_peaks(img, 4.5*std_noise, 8, center_int)
    source_finder = DAOStarFinder(fwhm=7.9, threshold=4.5*std_noise)
    sources = source_finder(np.squeeze(img))
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 5))

    axs[0].imshow(img, vmin=0, vmax=30/1000, origin='lower', cmap='viridis', aspect='equal')
    axs[1].imshow(img, vmin=0, vmax=30/1000, origin='lower', cmap='viridis', aspect='equal')

    axs[0].scatter(supressed_peaks[:, 0], supressed_peaks[:, 1], color='red', s=4, marker='o')
    axs[1].scatter(positions[:, 0], positions[:, 1], color='red', s=4, marker='o')

    plt.show()
    fig.savefig(r"D:\Sterrenkunde\Master Thesis\Code\MWE\sourcedetection.png", dpi=400)
    plt.close(fig)


# def confusion_score(Target_catalog, Reconstructed_catalog, confusion_df, rnd_its, matching_args):
#     confusion_df_coincidence = confusion_df.copy()
#     # Completeness
#     for ImageID in np.unique(Target_catalog["ImageID"]):
#         mask_target = Target_catalog.iloc[:, -1] == ImageID
#         mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
#         if len(mask_generated) > 0:
#             if np.sum(mask_generated) > 0: # For computational efficiency, do this after the first check
#                 for target_source_idx, target_source in Target_catalog[mask_target].iterrows():
#                     r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
#                     rmin_idx = np.argmin(r.values)
#                     #print(Reconstructed_catalog[mask_generated]['peak'])
#                     if r.values[rmin_idx] <= matching_args["max_distance"]: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
#                         for idx, bin in enumerate(confusion_df['Flux bins']):
#                             if bin[0] <= target_source["peak"] < bin[1]:
#                                 confusion_df.loc[idx, 'TPc'] += 1;
#                     else:
#                         for idx, bin in enumerate(confusion_df['Flux bins']):
#                             if bin[0] <= target_source["peak"] < bin[1]:
#                                 confusion_df.loc[idx, 'FNc'] += 1;
    
#     # Reliability
#     for ImageID in np.unique(Reconstructed_catalog["ImageID"]):
#         mask_target = Target_catalog.iloc[:, -1] == ImageID
#         mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        
#         if len(mask_target) > 0:
#             for gen_source_idx, gen_source in Reconstructed_catalog[mask_generated].iterrows():
#                 r = np.sqrt((Target_catalog[mask_target]['xpix'] - gen_source['xpix'])**2 + (Target_catalog[mask_target]['ypix'] - gen_source['ypix'])**2)
#                 rmin_idx = np.argmin(r.values)
#                 if r.values[rmin_idx] <= matching_args["max_distance"]:
#                     for idx, bin in enumerate(confusion_df['Flux bins']):
#                         if bin[0] <= gen_source['peak'] < bin[1]:
#                             confusion_df.loc[idx, 'TPr'] += 1;
#                 else:
#                     for idx, bin in enumerate(confusion_df['Flux bins']):
#                         if bin[0] <= gen_source['peak'] < bin[1]:
#                             confusion_df.loc[idx, 'FPr'] += 1;
    
#     for i in tqdm(range(rnd_its), desc="Computing the fake 2d distribution for Completeness and Reliability computations..."):
#         Target_catalog_copy = Target_catalog.copy()
#         Target_catalog_copy[["xpix", "ypix"]] += np.random.uniform(low=-matching_args["max_offset_pixels"], high=matching_args["max_offset_pixels"], size=Target_catalog[["xpix", "ypix"]].shape)
#         # Completeness
#         for ImageID in np.unique(Target_catalog_copy["ImageID"]):
#             mask_target = Target_catalog_copy.iloc[:, -1] == ImageID
#             mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
#             if len(mask_generated) > 0:
#                 if np.sum(mask_generated) > 0: # For computational efficiency, do this after the first check
#                     for target_source_idx, target_source in Target_catalog_copy[mask_target].iterrows():
#                         r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
#                         rmin_idx = np.argmin(r.values)
#                         #print(Reconstructed_catalog[mask_generated]['peak'])
#                         if r.values[rmin_idx] <= matching_args["max_distance"]: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
#                             for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
#                                 if bin[0] <= target_source["peak"] < bin[1]:
#                                     confusion_df_coincidence.loc[idx, 'TPc'] += 1;
#                         else:
#                             for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
#                                 if bin[0] <= target_source["peak"] < bin[1]:
#                                     confusion_df_coincidence.loc[idx, 'FNc'] += 1;

#         # Reliability
#         for ImageID in np.unique(Reconstructed_catalog["ImageID"]):
#             mask_target = Target_catalog_copy.iloc[:, -1] == ImageID
#             mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
            
#             if len(mask_target) > 0:
#                 for gen_source_idx, gen_source in Reconstructed_catalog[mask_generated].iterrows():
#                     r = np.sqrt((Target_catalog_copy[mask_target]['xpix'] - gen_source['xpix'])**2 + (Target_catalog_copy[mask_target]['ypix'] - gen_source['ypix'])**2)
#                     rmin_idx = np.argmin(r.values)
#                     if r.values[rmin_idx] <= matching_args["max_distance"]:
#                         for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
#                             if bin[0] <= gen_source['peak'] < bin[1]:
#                                 confusion_df_coincidence.loc[idx, 'TPr'] += 1;
#                     else:
#                         for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
#                             if bin[0] <= gen_source['peak'] < bin[1]:
#                                 confusion_df_coincidence.loc[idx, 'FPr'] += 1;
    
#     # Subtract the mean of the coincidence distribution from potential distribution of True positives
#     confusion_df[['TPc', 'TPr']] -= confusion_df_coincidence[['TPc', 'TPr']]/rnd_its

#     # Add the mean of the coincidence distribution to the potential distribution of false positives and false negatives
#     confusion_df[['FNc', 'FPr']] += confusion_df_coincidence[['TPc', 'TPr']]/rnd_its
    
#     # Calculate completeness and reliability of test sample
#     ## If needed resolve zero-occurences
#     for i in range(len(confusion_df['Flux bins'])):
#         if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
#             confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])
    
#         if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
#             confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
#     return confusion_df

def comp_completeness(Target_catalog, Reconstructed_catalog, confusion_df, matching_args):
    # Completeness
    for ImageID in np.unique(Target_catalog["ImageID"]):
        mask_target = Target_catalog.iloc[:, -1] == ImageID
        mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        if len(mask_generated) > 0:
            if np.sum(mask_generated) > 0: # For computational efficiency, do this after the first check
                for target_source_idx, target_source in Target_catalog[mask_target].iterrows():
                    r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
                    rmin_idx = np.argmin(r.values)
                    #print(Reconstructed_catalog[mask_generated]['peak'])
                    if r.values[rmin_idx] <= matching_args["max_distance"]: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                        for idx, bin in enumerate(confusion_df['Flux bins']):
                            if bin[0] <= target_source["peak"] < bin[1]:
                                confusion_df.loc[idx, 'TPc'] += 1;
                    else:
                        for idx, bin in enumerate(confusion_df['Flux bins']):
                            if bin[0] <= target_source["peak"] < bin[1]:
                                confusion_df.loc[idx, 'FNc'] += 1;
    return confusion_df


def comp_reliability(Target_catalog, Reconstructed_catalog, confusion_df, matching_args):
    # Reliability
    for ImageID in np.unique(Reconstructed_catalog["ImageID"]):
        mask_target = Target_catalog.iloc[:, -1] == ImageID
        mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        
        if len(mask_target) > 0:
            for gen_source_idx, gen_source in Reconstructed_catalog[mask_generated].iterrows():
                r = np.sqrt((Target_catalog[mask_target]['xpix'] - gen_source['xpix'])**2 + (Target_catalog[mask_target]['ypix'] - gen_source['ypix'])**2)
                rmin_idx = np.argmin(r.values)
                if r.values[rmin_idx] <= matching_args["max_distance"]:
                    for idx, bin in enumerate(confusion_df['Flux bins']):
                        if bin[0] <= gen_source['peak'] < bin[1]:
                            confusion_df.loc[idx, 'TPr'] += 1;
                else:
                    for idx, bin in enumerate(confusion_df['Flux bins']):
                        if bin[0] <= gen_source['peak'] < bin[1]:
                            confusion_df.loc[idx, 'FPr'] += 1;
    return confusion_df
    
# def confusion_score(Target_catalog, Reconstructed_catalog, confusion_df, rnd_its, matching_args):
#     confusion_df_copy1 = confusion_df.copy()
#     confusion_df_copy2 = confusion_df.copy()
#     confusion_df_coincidence_copy1 = confusion_df.copy()
#     confusion_df_coincidence_copy2 = confusion_df.copy()
#     confusion_df_coincidence = confusion_df.copy()
#     with Pool(2) as p:
#         result_completeness = p.apply_async(comp_completeness, args=(Target_catalog, Reconstructed_catalog, confusion_df_copy1, matching_args))
#         result_reliability = p.apply_async(comp_reliability, args=(Target_catalog, Reconstructed_catalog, confusion_df_copy2, matching_args))

#         confusion_df_copy1 = result_completeness.get()
#         confusion_df_copy2 = result_reliability.get()

#         # Add results to confusion_df
#         confusion_df.loc[:, ['TPc', 'FNc']] = confusion_df_copy1.loc[:, ['TPc', 'FNc']]
#         confusion_df.loc[:, ['TPr', 'FPr']] = confusion_df_copy2.loc[:, ['TPr', 'FPr']]

#     for i in tqdm(range(rnd_its), desc="Computing the fake 2d distribution for Completeness and Reliability..."):
#         Target_catalog_copy = Target_catalog.copy()
#         Target_catalog_copy[["xpix", "ypix"]] += np.random.uniform(low=-matching_args["max_offset_pixels"], high=matching_args["max_offset_pixels"], size=Target_catalog[["xpix", "ypix"]].shape)

#         with Pool(2) as p:
#             result_completeness = p.apply_async(comp_completeness, args=(Target_catalog, Reconstructed_catalog, confusion_df_coincidence_copy1, matching_args))
#             result_reliability = p.apply_async(comp_reliability, args=(Target_catalog, Reconstructed_catalog, confusion_df_coincidence_copy2, matching_args))

#             confusion_df_coincidence_copy1 = result_completeness.get()
#             confusion_df_coincidence_copy2 = result_reliability.get()

#             # Add results to confusion_df
#             confusion_df_coincidence.loc[:, ['TPc', 'FNc']] = confusion_df_coincidence_copy1.loc[:, ['TPc', 'FNc']]
#             confusion_df_coincidence.loc[:, ['TPr', 'FPr']] = confusion_df_coincidence_copy2.loc[:, ['TPr', 'FPr']]

#     print("coincidence df", confusion_df_coincidence)
#     print("confusion_1", confusion_df)


#     confusion_df.loc[:, ["TPc", "TPr"]] -= confusion_df_coincidence.loc[:, ['TPc', 'TPr']] / rnd_its
#     confusion_df.loc[:, ["FNc", "FPr"]] += confusion_df_coincidence.loc[:, ['TPc', 'TPr']].values / rnd_its

#     print("confusion_2", confusion_df)

#     # Calculate completeness and reliability of test sample
#     ## If needed resolve zero-occurences
#     for i in range(len(confusion_df['Flux bins'])):
#         if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
#             confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])
    
#         if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
#             confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
#             confusion_df.loc[i, 'flag_R'] = confusion_df.loc[i, 'flag_TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
#     print("confusion_3", confusion_df)
#     return confusion_df


def confusion_score(Target_catalog, Reconstructed_catalog, confusion_df, rnd_its, matching_args):
    confusion_df_coincidence = confusion_df.copy()
    # Completeness
    for ImageID in tqdm(np.unique(Target_catalog["ImageID"]), desc="Completeness..."):
        mask_target = Target_catalog.iloc[:, -1] == ImageID
        mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        if len(mask_generated) > 0:
            if np.sum(mask_generated) > 0: # For computational efficiency, do this after the first check
                for target_source_idx, target_source in Target_catalog[mask_target].iterrows():
                    r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
                    rmin_idx = np.argmin(r.values)
                    #print(Reconstructed_catalog[mask_generated]['peak'])
                    if r.values[rmin_idx] <= matching_args["max_distance"]: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                        for idx, bin in enumerate(confusion_df['Flux bins']):
                            if bin[0] <= target_source["peak"] < bin[1]:
                                confusion_df.loc[idx, 'TPc'] += 1;
                    else:
                        for idx, bin in enumerate(confusion_df['Flux bins']):
                            if bin[0] <= target_source["peak"] < bin[1]:
                                confusion_df.loc[idx, 'FNc'] += 1;
    # Reliability
    for ImageID in tqdm(np.unique(Reconstructed_catalog["ImageID"]), desc="Reliability..."):
        mask_target = Target_catalog.iloc[:, -1] == ImageID
        mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        
        if len(mask_target) > 0:
            for gen_source_idx, gen_source in Reconstructed_catalog[mask_generated].iterrows():
                r = np.sqrt((Target_catalog[mask_target]['xpix'] - gen_source['xpix'])**2 + (Target_catalog[mask_target]['ypix'] - gen_source['ypix'])**2)
                rmin_idx = np.argmin(r.values)
                if r.values[rmin_idx] <= matching_args["max_distance"]:
                    for idx, bin in enumerate(confusion_df['Flux bins']):
                        if bin[0] <= gen_source['peak'] < bin[1]:
                            confusion_df.loc[idx, 'TPr'] += 1;
                else:
                    for idx, bin in enumerate(confusion_df['Flux bins']):
                        if bin[0] <= gen_source['peak'] < bin[1]:
                            confusion_df.loc[idx, 'FPr'] += 1;
    # for i in tqdm(range(rnd_its), desc="Computing the fake 2d distribution for Completeness and Reliability..."):
    #     Target_catalog_copy = Target_catalog.copy()
    #     Target_catalog_copy[["xpix", "ypix"]] += np.random.uniform(low=-matching_args["max_offset_pixels"], high=matching_args["max_offset_pixels"], size=Target_catalog[["xpix", "ypix"]].shape)
    #     # Completeness
    #     for ImageID in np.unique(Target_catalog_copy["ImageID"]):
    #         mask_target = Target_catalog_copy.iloc[:, -1] == ImageID
    #         mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
    #         if len(mask_generated) > 0:
    #             if np.sum(mask_generated) > 0: # For computational efficiency, do this after the first check
    #                 for target_source_idx, target_source in Target_catalog_copy[mask_target].iterrows():
    #                     r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
    #                     rmin_idx = np.argmin(r.values)
    #                     #print(Reconstructed_catalog[mask_generated]['peak'])
    #                     if r.values[rmin_idx] <= matching_args["max_distance"]: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
    #                         for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
    #                             if bin[0] <= target_source["peak"] < bin[1]:
    #                                 confusion_df_coincidence.loc[idx, 'TPc'] += 1;
    #                     else:
    #                         for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
    #                             if bin[0] <= target_source["peak"] < bin[1]:
    #                                 confusion_df_coincidence.loc[idx, 'FNc'] += 1;

    #     # Reliability
    #     for ImageID in np.unique(Reconstructed_catalog["ImageID"]):
    #         mask_target = Target_catalog_copy.iloc[:, -1] == ImageID
    #         mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
            
    #         if len(mask_target) > 0:
    #             for gen_source_idx, gen_source in Reconstructed_catalog[mask_generated].iterrows():
    #                 r = np.sqrt((Target_catalog_copy[mask_target]['xpix'] - gen_source['xpix'])**2 + (Target_catalog_copy[mask_target]['ypix'] - gen_source['ypix'])**2)
    #                 rmin_idx = np.argmin(r.values)
    #                 if r.values[rmin_idx] <= matching_args["max_distance"]:
    #                     for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
    #                         if bin[0] <= gen_source['peak'] < bin[1]:
    #                             confusion_df_coincidence.loc[idx, 'TPr'] += 1;
    #                 else:
    #                     for idx, bin in enumerate(confusion_df_coincidence['Flux bins']):
    #                         if bin[0] <= gen_source['peak'] < bin[1]:
    #                             confusion_df_coincidence.loc[idx, 'FPr'] += 1;

    # print("coincidence df", confusion_df_coincidence)
    # print("confusion_1", confusion_df)


    # confusion_df.loc[:, ["TPc", "TPr"]] -= np.round(confusion_df_coincidence.loc[:, ['TPc', 'TPr']].values / rnd_its)
    # confusion_df.loc[:, ["FNc", "FPr"]] += np.round(confusion_df_coincidence.loc[:, ['TPc', 'TPr']].values / rnd_its)

    # print("confusion_2", confusion_df)
    # # Set negative values to 0
    # cols_to_check = ['TPc', 'TPr', 'FNc', 'FPr']
    # for col in cols_to_check:
    #     confusion_df.loc[confusion_df[col] < 0, col] = 0

    # Calculate completeness and reliability of test sample
    ## If needed resolve zero-occurences
    for i in range(len(confusion_df['Flux bins'])):
        if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
            confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])
    
        if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
            confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
    
    return confusion_df

def fit_flux_distribution(blind_matches_catalog, rnd_its, *catalogs, **matching_args):
    cols = blind_matches_catalog.columns.tolist() if isinstance(blind_matches_catalog, pd.DataFrame) else blind_matches_catalog.keys()

    # Create the 2D distribution plot
    bins = 50
    xbins = np.logspace(np.log10(1), np.log10(150), bins + 1)/1000
    ybins = np.logspace(np.log10(1), np.log10(150), bins + 1)/1000
    H_blind, xedges_blind, yedges_blind = np.histogram2d(blind_matches_catalog[cols[0]], blind_matches_catalog[cols[1]], bins=(xbins, ybins))

    mask = np.ma.masked_where(H_blind.T == 0, H_blind.T)

    x_centers = (xbins[:-1] + xbins[1:]) / 2 * 1000
    y_centers = (ybins[:-1] + ybins[1:]) / 2 * 1000
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute the fake 2d distribution
    H_rnd = np.zeros((bins, bins)) # H has x=rows and y=columns, H.T is the transpose
    for i in range(rnd_its):
        rnd_matches_catalog = Coincidence_Matches(*catalogs, return_df=True, **matching_args)
        H_it, xedges, yedges = np.histogram2d(rnd_matches_catalog[cols[0]], rnd_matches_catalog[cols[1]], bins=(xbins, ybins))

        H_rnd += H_it/rnd_its

    mask = np.ma.masked_where(H_rnd.T == 0, H_rnd.T)

    # Compute the 2D distribution plot of good matches
    H = np.round(H_blind - H_rnd, 0).astype(np.int32)
    H[H<= 0] = 0
    mask = np.ma.masked_where(H.T == 0, H.T)
    # Profile function to fit the 2D distribution plot
    func = lambda x, a, b: a*x + b
    
    Target_flux_bins = []
    Reconstructed_flux_bins = []
    for row_idx, row in enumerate(H.T): # y
        for col_idx, counts in enumerate(row): # x
            if counts > 0:
                for j in range(int(counts)):
                    Reconstructed_flux_bins.append(Y[row_idx, col_idx])
                    Target_flux_bins.append(X[row_idx, col_idx])

    popt, pcov = curve_fit(func, np.array(Target_flux_bins), np.array(Reconstructed_flux_bins), bounds=([0, -30], [2, 30]))
        
    return popt[0], popt[1], Target_flux_bins, Reconstructed_flux_bins


def fill_pixcatalog(img_batch, cat, ImageIDList, instr_noise):
    # Squeeze to avoid unnecessary computation of the image shape
    img_batch = np.squeeze(img_batch)      
    source_finder = DAOStarFinder(fwhm=7.9, threshold=instr_noise)

    # Fill generated catalog
    for i in range(img_batch.shape[0]):
        sources = source_finder(img_batch[i])
        # Make sure that there are detectable sources in the image
        try:
            tr = np.transpose((sources['xcentroid'], sources['ycentroid']))
        except:
            continue

        for source in sources:
            cat["peak"].append(source["peak"])
            cat["xpix"].append(source["xcentroid"])
            cat["ypix"].append(source["ycentroid"])
            cat["ImageID"].append(ImageIDList[i])
        
    return cat

def find_matches_parallel(Target_catalog, Reconstructed_catalog, ImageIDList, max_distance=20, ReproductionRatio_min=0.5, ReproductionRatio_max=1.5):
    matches_catalog = {"Target Source Flux": [], "Reconstructed Source Flux": [], "Distance" : []}
    ## Iterate over target catalog
    ## Find the best match 
    ## Fill the matches catalog
    for ImageID in ImageIDList:
        mask_target = Target_catalog.iloc[:, -1] == ImageID
        mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        if len(mask_generated) > 0:
            if np.sum(mask_generated) > 0:
                for target_source_idx, target_source in Target_catalog[mask_target].iterrows():
                    if len(mask_generated) != 0:
                        r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source.loc['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
                        rmin_idx = np.argmin(r.values)
                        #print(Reconstructed_catalog[mask_generated]['peak'])
                        if r.values[rmin_idx] <= max_distance: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                            matches_catalog["Target Source Flux"].append(target_source['peak'])
                            matches_catalog["Reconstructed Source Flux"].append(Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx])
                            matches_catalog["Distance"].append(r.values[rmin_idx])
    return matches_catalog

def find_matches(Target_catalog, Reconstructed_catalog, max_distance=20, ReproductionRatio_min=0.5, ReproductionRatio_max=1.5, return_df=False):   
    ImageIDList = np.unique(Target_catalog["ImageID"])
    ImageIDList1 = np.unique(Target_catalog["ImageID"])[:len(ImageIDList)//3]
    ImageIDList2 = np.unique(Target_catalog["ImageID"])[len(ImageIDList)//3:len(ImageIDList)//3 * 2]
    ImageIDList3 = np.unique(Target_catalog["ImageID"])[len(ImageIDList)//3 * 2:]
    with Pool(3) as p:
        result1 = p.apply_async(find_matches_parallel, args=(Target_catalog, Reconstructed_catalog, ImageIDList1, max_distance, ReproductionRatio_min, ReproductionRatio_max))
        result2 = p.apply_async(find_matches_parallel, args=(Target_catalog, Reconstructed_catalog, ImageIDList2, max_distance, ReproductionRatio_min, ReproductionRatio_max))
        result3 = p.apply_async(find_matches_parallel, args=(Target_catalog, Reconstructed_catalog, ImageIDList3, max_distance, ReproductionRatio_min, ReproductionRatio_max))
        matches_catalog_1 = result1.get()
        matches_catalog_2 = result2.get()
        matches_catalog_3 = result3.get()

        matches_catalog = {column: matches_catalog_1[column] + matches_catalog_2.get(column, []) + matches_catalog_3.get(column, []) for column in matches_catalog_1}
    if return_df:
        cols = ["Target Source Flux", "Reconstructed Source Flux", "Distance"]
        return pd.DataFrame(matches_catalog, columns=cols)
    else:
        return matches_catalog
    
def comp_completeness_n_matches_parallel(Target_catalog, Reconstructed_catalog, ImageIDList, confusion_df, matching_args):
    matches_catalog = {"Target Source Flux": [], "Reconstructed Source Flux": [], "Distance" : []}
    for ImageID in ImageIDList:
        mask_target = Target_catalog.iloc[:, -1] == ImageID
        mask_generated = Reconstructed_catalog.iloc[:, -1] == ImageID
        if len(mask_generated) > 0:
            if np.sum(mask_generated) > 0:
                for target_source_idx, target_source in Target_catalog[mask_target].iterrows():
                    if len(mask_generated) != 0:
                        r = np.sqrt((target_source['xpix'] - Reconstructed_catalog[mask_generated]['xpix'])**2 + (target_source.loc['ypix'] - Reconstructed_catalog[mask_generated]['ypix'])**2)
                        rmin_idx = np.argmin(r.values)
                        if r.values[rmin_idx] <= matching_args["max_distance"]: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                            matches_catalog["Target Source Flux"].append(target_source['peak'])
                            matches_catalog["Reconstructed Source Flux"].append(Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx])
                            matches_catalog["Distance"].append(r.values[rmin_idx])
                            for idx, bin in enumerate(confusion_df['Flux bins']):
                                if bin[0] <= target_source["peak"] < bin[1]:
                                    confusion_df.loc[idx, 'TPc'] += 1;
                        else:
                            for idx, bin in enumerate(confusion_df['Flux bins']):
                                if bin[0] <= target_source["peak"] < bin[1]:
                                    confusion_df.loc[idx, 'FNc'] += 1;
    return confusion_df, matches_catalog

   
def get_matches_n_confusion(Target_catalog, Reconstructed_catalog, confusion_df, matching_args, return_matches_df=False):
    confusion_df_r_copy1 = confusion_df.copy()

    confusion_df_c_copy1 = confusion_df.copy()
    confusion_df_c_copy2 = confusion_df.copy()

    ImageIDList = np.unique(Target_catalog["ImageID"])
    ImageIDList1 = np.unique(Target_catalog["ImageID"])[:len(ImageIDList)//2]
    ImageIDList2 = np.unique(Target_catalog["ImageID"])[len(ImageIDList)//2:]

    with Pool(3) as p:
        result_reliability = p.apply_async(comp_reliability, args=(Target_catalog, Reconstructed_catalog, confusion_df_r_copy1, matching_args))

        result1_completeness_n_matches = p.apply_async(comp_completeness_n_matches_parallel, args=(Target_catalog, Reconstructed_catalog, ImageIDList1, confusion_df_c_copy1, matching_args))
        result2_completeness_n_matches = p.apply_async(comp_completeness_n_matches_parallel, args=(Target_catalog, Reconstructed_catalog, ImageIDList2, confusion_df_c_copy2, matching_args))

        confusion_df_r_copy1 = result_reliability.get()
        confusion_df_c_copy1, matches_catalog_1 = result1_completeness_n_matches.get()
        confusion_df_c_copy2, matches_catalog_2 = result2_completeness_n_matches.get()

        matches_catalog = {column: matches_catalog_1[column] + matches_catalog_2.get(column, []) for column in matches_catalog_1}

        # Add completeness results to confusion df
        confusion_df.loc[:, ['TPc', 'FNc']] += confusion_df_c_copy1.loc[:, ['TPc', 'FNc']]
        confusion_df.loc[:, ['TPc', 'FNc']] += confusion_df_c_copy2.loc[:, ['TPc', 'FNc']]

        # Add reliability results to confusion_df
        confusion_df.loc[:, ['TPr', 'FPr']] = confusion_df_r_copy1.loc[:, ['TPr', 'FPr']]

    # Calculate completeness and reliability of test sample
    ## If needed resolve zero-occurences
    for i in range(len(confusion_df['Flux bins'])):
        if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
            confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])
    
        if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
            confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
            confusion_df.loc[i, 'flag_R'] = confusion_df.loc[i, 'flag_TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])


    if return_matches_df:
        cols = ["Target Source Flux", "Reconstructed Source Flux", "Distance"]
        return confusion_df, pd.DataFrame(matches_catalog, columns=cols)
    else:
        return confusion_df, matches_catalog
    
def Coincidence_Matches(catalogs, return_df=False, **matching_args):
    Target_catalog, Reconstructed_catalog = catalogs
    
    Target_catalog_copy = Target_catalog.copy()
    # Mutate the target catalog x and y coordinates by a random offset
    Target_catalog_copy[["xpix", "ypix"]] += np.random.uniform(low=-matching_args["max_offset_pixels"], high=matching_args["max_offset_pixels"], size=Target_catalog[["xpix", "ypix"]].shape)
    
    # Perform random Blind Source Matching  
    # The resulting catalog contains fake matches
    matching_args.pop("max_offset_pixels", None) # Remove max_offset_pixels from matching_args
    rnd_matches_catalog = find_matches(Target_catalog_copy, Reconstructed_catalog, **matching_args, return_df=return_df)

    if return_df:
        cols = ["Target Source Flux", "Reconstructed Source Flux", "Distance"]
        return pd.DataFrame(rnd_matches_catalog, columns=cols)
    else:
        return rnd_matches_catalog

