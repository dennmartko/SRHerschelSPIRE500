######################
###     IMPORTS    ###
######################

# Standard library imports
import os
import configparser
import gc
import time
import random

# Third-party library imports
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from astroML import correlation as corr

# Custom module imports
from PlotLib.PlotFunctionsTest import *
from metric_funcs import *

# TensorFlow import
import tensorflow as tf
tf.random.set_seed(42)

#######################
###    TEST CLASS   ###
#######################

class SRTesterGPU():
    def __init__(self, path_to_config) -> None:
        # Load configuration file for training
        self.config = configparser.ConfigParser()
        self.config.read(path_to_config)

        # Create any missing directories and initialize necessary parameters
        self.DIM = (424, 424)

        # Instrumental Noise
        self.instr_noise =  2.0/1000 #Jy/beam

        ## Paths with train data
        self.path_test = self.config['COMMON']['path_test'].rstrip().lstrip()
        # self.correlation_data_path = correlation_data_path 
        ## Indicate purpose of run
        self.RUN_NAME = self.config['MODELS']['model_name'].rstrip().lstrip()

        ## Load path to models dir
        self.models_lib_path = self.config['COMMON']['model_outdir'].rstrip().lstrip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)
        self.kind = self.config['MODELS']['model_type'].rstrip().lstrip()

        ## Paths to catalog
        self.path_cat = self.config['COMMON']['path_catalog'].rstrip().lstrip()

        ## Set classes
        self.classes = [i.strip(' ') for i in self.config['COMMON']['input'].rstrip().lstrip().split(",")] + [self.config['COMMON']['target'].rstrip().lstrip()] # Inp first, target last

        self.TOTAL_SAMPLES = len([entry for entry in os.listdir(os.path.join(self.path_test, self.classes[0]))])
        # self.TOTAL_SAMPLES_CORR = len([entry for entry in os.listdir(os.path.join(self.correlation_data_path, self.classes[0]))])
        self.tdir_out = os.path.join(self.model_path, f"results_{self.kind}")

        if not os.path.isdir(self.tdir_out):
            os.mkdir(self.tdir_out)

    def LoadTestData(self):
        self.TEST_BATCH_SIZE = 6

        self.test_arr_X = np.zeros((self.TOTAL_SAMPLES, 4, 106, 106))
        self.test_arr_Y = np.zeros((self.TOTAL_SAMPLES, 1, 424, 424))

        self.wcs_arr = []
        self.wcs_arr_plw = []
        for i in tqdm(range(self.TOTAL_SAMPLES), desc=f"Loading Data From {self.path_test}"):
            for k in range(len(self.classes)):
                with fits.open(os.path.join(self.path_test, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
                    if k == len(self.classes) - 1:
                        self.test_arr_Y[i] = hdu[0].data
                        self.wcs_arr.append(WCS(hdu[0].header))
                        arr = np.array([list(row) for row in hdu[1].data])
                        if i == 0:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = arr.copy()
                        else:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = np.vstack((self.target_image_sources_cat_test, arr))

                        del arr;
                    elif self.classes[k] == "500":
                        self.test_arr_X[i][k] = hdu[0].data
                        self.wcs_arr_plw.append(WCS(hdu[0].header))
                        arr = np.array([list(row) for row in hdu[1].data])
                        if i == 0:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.plw_image_sources_cat_test = arr.copy()
                        else:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.plw_image_sources_cat_test = np.vstack((self.plw_image_sources_cat_test, arr))

                        del arr;
                    else:
                        self.test_arr_X[i][k] = hdu[0].data   

        self.dtypes = {"peak": np.float32, "xpix": np.float32, "ypix":np.float32, "ra":np.float32, "dec":np.float32, "ImageID":np.int32}
        self.cat_cols = ["peak", "xpix", "ypix", "ra", "dec", "ImageID"]
        self.target_catalog = pd.DataFrame(data=self.target_image_sources_cat_test, columns=self.cat_cols).astype(self.dtypes)
        self.plw_catalog = pd.DataFrame(data=self.plw_image_sources_cat_test, columns=self.cat_cols).astype(self.dtypes)
        del self.target_image_sources_cat_test;
        del self.plw_image_sources_cat_test;
        # Load in the Simulation master catalog
        self.sim_master_cat = pd.DataFrame(fits.getdata(self.path_cat)).astype('<f8')
        self.sim_master_cat = self.sim_master_cat[self.sim_master_cat["SSPIRE500"] >= 2/1000]
        # Free memory
        gc.collect()

    def LoadModel(self, kind, model_path):
        self.generator = tf.keras.models.load_model(os.path.join(model_path, f'{kind}_Model'))
        gc.collect()

    def fill_worldcatalog(self, img_batch, cat, ImageIDList, instr_noise):
        img_batch = np.squeeze(img_batch)      
        source_finder = DAOStarFinder(fwhm=7.9, threshold=instr_noise)

        # Fill generated catalog
        for i in range(img_batch.shape[0]):
            sources = source_finder(img_batch[i])
            # Make sure that there are detectable sources in the image
            try:
                tr = np.transpose(np.vstack((sources['xcentroid'], sources['ycentroid'])))
                tr_world = self.wcs_arr[ImageIDList[i]].wcs_pix2world(tr, 0)
            except:
                continue

            for idx, source in enumerate(sources):
                cat["peak"].append(source["peak"])
                cat["xpix"].append(source["xcentroid"])
                cat["ypix"].append(source["ycentroid"])
                cat["ra"].append(tr_world[idx][0])
                cat["dec"].append(tr_world[idx][1])
                cat["ImageID"].append(ImageIDList[i])
        return cat


    def match_with_master_catalog(self, catalog, max_distance=4, ReproductionRatio_min=0.5, ReproductionRatio_max=2.5, return_df=False):
        matches_catalog = {"Master Catalog Source Flux": [], "Detected Source Flux": [], "Distance" : []}
        # Pre-append function boosts iteration speed 1it/s
        detsflux_app = matches_catalog["Detected Source Flux"].append
        mastercatsflux_app = matches_catalog["Master Catalog Source Flux"].append
        distance_app = matches_catalog["Distance"].append
        # pre-load the ra and dec coordinates to prevvent repeated lookups
        master_ra = self.sim_master_cat['ra'].values
        master_dec = self.sim_master_cat['dec'].values
        master_flux = self.sim_master_cat['SSPIRE500'].values

        ## Iterate over target catalog
        ## Find the best match
        ## Fill the matches catalog
        for detected_source_idx, detected_source in tqdm(catalog.iterrows(), desc="matching with master catalog...", total=catalog.shape[0]):
            r = np.sqrt((detected_source['ra'] - master_ra)**2 + (detected_source['dec'] - master_dec)**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx]*3600 <= max_distance: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                detsflux_app(detected_source['peak'])
                mastercatsflux_app(master_flux[rmin_idx])
                distance_app(r[rmin_idx]*3600)

        if return_df:
            cols = ["Master Catalog Source Flux", "Detected Source Flux", "Distance"]
            return pd.DataFrame(matches_catalog, columns=cols)
        else:
            return matches_catalog

    # def match_with_master_catalog_parallel(self, catalog_chunk, max_distance):
    #     matches_catalog = {"Master Catalog Source Flux": [], "Detected Source Flux": [], "Distance" : []}

    #     for detected_source in tqdm(catalog_chunk.iterrows(), desc="Chunk matching with master catalog...", total=catalog_chunk.shape[0]):
    #         r = np.sqrt((detected_source['ra'] - self.sim_master_cat['ra'].values)**2 + (detected_source['dec'] - self.sim_master_cat['dec'].values)**2)
    #         rmin_idx = np.argmin(r)
    #         if r[rmin_idx]*3600 <= max_distance:
    #             matches_catalog["Detected Source Flux"].append(detected_source['peak'])
    #             matches_catalog["Master Catalog Source Flux"].append(self.sim_master_cat['SSPIRE500'].values[rmin_idx])
    #             matches_catalog["Distance"].append(r[rmin_idx]*3600)
    #     return matches_catalog

    # def match_with_master_catalog(self, catalog, max_distance=4, ReproductionRatio_min=0.5, ReproductionRatio_max=2.5, return_df=False):
    #     cpu_cores = 2  # Number of available CPU cores

    #     catalog_split = np.array_split(catalog, cpu_cores)

    #     with Pool(2) as p:
    #         result1 = p.apply_async(self.match_with_master_catalog_parallel, args=(catalog_split[0], max_distance))
    #         result2 = p.apply_async(self.match_with_master_catalog_parallel, args=(catalog_split[1], max_distance))
            
    #         matches_catalog1 = result1.get()
    #         matches_catalog2 = result2.get()

    #         matches_catalog = {column: matches_catalog1[column] + matches_catalog2.get(column, []) for column in matches_catalog1}

    #     if return_df:
    #         return pd.DataFrame(matches_catalog)
    #     else:
    #         return matches_catalog
    
    def plot_source_recovery(self, cat_matches, ylabel, save_path):
        plt.figure(figsize=(8,8))

        xmax = np.max(cat_matches["Master Catalog Source Flux"]*1000)
        true_line = np.linspace(0, 125, 100)

        bin_edges = np.logspace(np.log10(2), np.log10(np.max(cat_matches["Master Catalog Source Flux"]*1000)), 10, base=10)
        bin_indices_x = np.digitize(cat_matches["Master Catalog Source Flux"]*1000, bin_edges)
        bin_indices_y = np.digitize(cat_matches["Detected Source Flux"]*1000, bin_edges)

        bin_medians_x = [np.median((cat_matches["Master Catalog Source Flux"]*1000)[bin_indices_x == i]) for i in range(1, len(bin_edges))]
        bin_medians_y = [np.median((cat_matches["Detected Source Flux"]*1000)[bin_indices_x == i]) for i in range(1, len(bin_edges))]
        bin_std_errors_x = [np.std((cat_matches["Master Catalog Source Flux"]*1000)[bin_indices_x == i]) for i in range(1, len(bin_edges))]
        bin_std_errors_y = [np.std((cat_matches["Detected Source Flux"]*1000)[bin_indices_x == i]) for i in range(1, len(bin_edges))]

        plt.scatter(cat_matches["Master Catalog Source Flux"]*1000, cat_matches["Detected Source Flux"]*1000, s=0.5, marker='o', color='blue', label="Matched Sources")
        plt.plot(true_line, true_line, color='red', label="1:1 Recovery", linestyle='dotted', linewidth=2)
        plt.errorbar(bin_medians_x, bin_medians_y, xerr=bin_std_errors_x, yerr=bin_std_errors_y, fmt='.', color='red', label='Median with SE', markersize=5, capsize=3)

        plt.xlabel(r"Input $500 \mu m$ Source Flux $S_{in}$ [mJy]", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        plt.yscale('log')
        plt.xscale('log')

        plt.xlim([1, 125])
        plt.ylim([1, 125])

        plt.legend(fontsize=12)
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        plt.close()

    def evaluate_custom_source_detection(self, img, window_size, kernel_size, save):
        std_noise = 2./1000 #mJy
        center_int = tf.cast(tf.round((kernel_size - 1)/2), tf.int32)
        peaks, supressed_peaks = find_local_peaks(img, std_noise, 8, center_int)
        source_finder = DAOStarFinder(fwhm=7.9, threshold=std_noise)
        sources = source_finder(np.squeeze(img))
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        fig, axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 5))

        axs[0].imshow(img, vmin=0, vmax=20/1000, origin='lower', cmap='viridis', aspect='equal')
        axs[1].imshow(img, vmin=0, vmax=20/1000, origin='lower', cmap='viridis', aspect='equal')

        axs[0].scatter(supressed_peaks[:, 0], supressed_peaks[:, 1], marker='o', facecolors='none', edgecolors='red', s=30,)
        axs[1].scatter(positions[:, 0], positions[:, 1], marker='o', facecolors='none', edgecolors='red', s=30,)
        axs[0].set_title("Custom Source Detection")
        axs[1].set_title("DAOFIND algorithm (Photutils)")

        fig.savefig(save, dpi=400, bbox_inches='tight')
        plt.close(fig)

    def Analysis(self):
        self.LoadModel(kind=self.kind, model_path=self.model_path)
        # Generated Source Catalog needs to be filled first  
        reconstructed_catalog = {"peak": [], "xpix": [], "ypix": [], "ra":[], "dec":[], "ImageID": []}
        its = self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE if self.test_arr_X.shape[0] > self.TEST_BATCH_SIZE else 1
        print(self.sim_master_cat)
        for batch_idx in tqdm(range(its), desc="Super-Resolving Test Data with main model!"):
            # Load the batch of data
            X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            Y = self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            
            # Compute the corresponding ImageIDs
            idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
            ImageIDList = idx_arr.tolist()

            # Super-Resolve the batch of test data
            gen_valid = self.generator(X, training=False).numpy()

            # Fill the generated catalog with detected sources
            reconstructed_catalog = self.fill_worldcatalog(gen_valid, reconstructed_catalog, ImageIDList, self.instr_noise)

        # Convert the generated catalog to a Pandas DataFrame
        reconstructed_catalog = pd.DataFrame(reconstructed_catalog, columns=self.cat_cols).astype(self.dtypes)

        self.evaluate_custom_source_detection(np.squeeze(gen_valid[0]), 8, 21, save=os.path.join(self.tdir_out, "custom_source_det_evaluation.png"))
        print(reconstructed_catalog[reconstructed_catalog["ImageID"] == 10])
        print(self.plw_catalog[self.plw_catalog["ImageID"] == 10])
        print(self.target_catalog[self.target_catalog["ImageID"] == 10])
        # Source-Catalog matching arguments
        ## max distance in degrees
        blind_matching_args = {"max_distance": 4, "ReproductionRatio_min": 0.5, "ReproductionRatio_max": 2.5}
        cat_SR_matches = self.match_with_master_catalog(reconstructed_catalog, **blind_matching_args, return_df=True)
        cat_plw_matches = self.match_with_master_catalog(self.plw_catalog, **blind_matching_args, return_df=True)
        cat_target_matches = self.match_with_master_catalog(self.target_catalog, **blind_matching_args, return_df=True)

        # evaluate custom source detection on one image


        # Create Scatter plots for each matched combination to illustrate super-resolution performance on test-test
        self.plot_source_recovery(cat_SR_matches, r"Super-Resolved $500\mu m$ Source Flux $S_{SR}$ [mJy]", os.path.join(self.tdir_out, "test_catalog_comparison_recovery_SR_matches.png"))
        self.plot_source_recovery(cat_plw_matches, r"Low Resolution $500\mu m$ Source Flux $S_{LR}$ [mJy]", os.path.join(self.tdir_out, "test_catalog_comparison_recovery_LR_matches.png"))
        self.plot_source_recovery(cat_target_matches, r"Target $500\mu m$ Source Flux $S_{target}$ [mJy]", os.path.join(self.tdir_out, "test_catalog_comparison_recovery_Y_matches.png"))

if __name__ == "__main__":
    SRModelTest = SRTesterGPU("compare_test_cat.ini")
    SRModelTest.LoadTestData()
    SRModelTest.Analysis()