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
    def __init__(self, path_to_config, gridmode = False, idx = None, MODEL2 = None) -> None:
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
        self.RUN_NAME = self.config['COMMON']['RUN_NAME'].rstrip().lstrip()
        if gridmode == True:
            self.RUN_NAME += f'_gridmode_{idx}'

        ## Load path to models dir
        self.models_lib_path = self.config['COMMON']['model_outdir'].rstrip().lstrip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)

        ## Set classes
        self.classes = [i.strip(' ') for i in self.config['COMMON']['input'].rstrip().lstrip().split(",")] + [self.config['COMMON']['target'].rstrip().lstrip()] # Inp first, target last

        self.TOTAL_SAMPLES = len([entry for entry in os.listdir(os.path.join(self.path_test, self.classes[0]))])
        # self.TOTAL_SAMPLES_CORR = len([entry for entry in os.listdir(os.path.join(self.correlation_data_path, self.classes[0]))])
        self.tdir_out = [os.path.join(self.model_path, "SCUBA2_results_BestFluxReproduction")] #[os.path.join(self.model_path, "results_BestFluxReproduction")] #[os.path.join(self.model_path, "SCUBA2_results_BestFluxReproduction")] 

        ## If we have another model path, ENABLE model comparison
        if MODEL2 is not None:
            self.model_path2 = os.path.join(self.models_lib_path, MODEL2)
            self.model_comparison = True
        else:
            self.model_comparison = False

        for tdir in self.tdir_out:
            if not os.path.isdir(tdir):
                os.mkdir(tdir)

        #Register GridMode
        self.gridmode = gridmode

    def LoadTestData(self):
        self.TEST_BATCH_SIZE = 6

        self.test_arr_X = np.zeros((self.TOTAL_SAMPLES, 4, 106, 106))
        self.test_arr_Y = np.zeros((self.TOTAL_SAMPLES, 1, 424, 424))

        # self.test_arr_X_corr = np.zeros((self.TOTAL_SAMPLES_CORR, 3, 106, 106))
        # self.test_arr_Y_corr = np.zeros((self.TOTAL_SAMPLES_CORR, 1, 424, 424))

        # self.test_wcs_corr = []

        for i in tqdm(range(self.TOTAL_SAMPLES), desc=f"Loading Data From {self.path_test}"):
            for k in range(len(self.classes)):
                with fits.open(os.path.join(self.path_test, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
                    if k == len(self.classes) - 1:
                        self.test_arr_Y[i] = hdu[0].data
                        arr = np.array([list(row[:-2]) for row in hdu[1].data])
                        if i == 0:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = arr.copy()
                        else:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = np.vstack((self.target_image_sources_cat_test, arr))

                        del arr;
                    else:
                        self.test_arr_X[i][k] = hdu[0].data   

        # for i in tqdm(range(self.TOTAL_SAMPLES_CORR), desc=f"Loading Data From {self.correlation_data_path}"):
        #     for k in range(len(self.classes)):
        #         with fits.open(os.path.join(self.correlation_data_path, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
        #             if k == len(self.classes) - 1:
        #                 self.test_arr_Y_corr[i] = hdu[0].data
        #                 self.test_wcs_corr.append(WCS(hdu[0].header))
        #             else:
        #                 self.test_arr_X_corr[i][k] = hdu[0].data  

        self.dtypes = {"peak": np.float32, "xpix": np.float32, "ypix":np.float32, "ImageID":np.int32}
        self.cat_cols = ["peak", "xpix", "ypix", "ImageID"]
        self.target_catalog = pd.DataFrame(data=self.target_image_sources_cat_test, columns=self.cat_cols).astype(self.dtypes)

        del self.target_image_sources_cat_test;
        # Free memory
        gc.collect()
    def LoadModel(self, kind, model_path):
        self.generator = tf.keras.models.load_model(os.path.join(model_path, f'{kind}_Model'))
        gc.collect()


    def TestAnalysis(self):
        kind = ["BestPSNR"]

        rnd_its = 10#100#100# Number of times to obtain a good estimate of distributions with randomness
        for idx, tdir in tqdm(enumerate(self.tdir_out), desc="Testing Models"):
            # Load correct model
            self.LoadModel(kind=kind[idx], model_path=self.model_path)

            # Generated Source Catalog needs to be filled first  
            reconstructed_catalog = {"peak": [], "xpix": [], "ypix": [], "ImageID": []}
            # first_block_idx = [i for i, layer in enumerate(self.generator.layers) if layer.name == 'tf.nn.softmax'][0]
            # # first_block = #[layer.output for layer in self.generator.layers[:first_block_idx+1]]
            # first_block_model = tf.keras.models.Model(inputs=self.generator.inputs, outputs=self.generator.layers[first_block_idx].output)
            its = self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE if self.test_arr_X.shape[0] > self.TEST_BATCH_SIZE else 1

            max_bin_value = 150 # mJy
            num_bins = 20
            flux_bin_edges = np.logspace(np.log10(1), np.log10(max_bin_value), num_bins + 1, base=10)/1000

            ## Zip the values array with itself shifted by one position to the left to create tuples of the left and right bounds of each bin
            self.flux_bins = list(zip(flux_bin_edges[:-1], flux_bin_edges[1:]))
            zero_list = np.zeros(len(self.flux_bins))

            # X = self.test_arr_X[0*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(1)].astype(np.float32)
            # print(first_block_model.predict(X).shape)
            # output_first_block = np.squeeze(first_block_model.predict(X))[0]
            # plt.figure(figsize=(12,12))
            # for i in range(1):
            #     plt.subplot(1, 1, i+1)
            #     plt.imshow(output_first_block, cmap="viridis")
            #     plt.axis('off')
            # plt.tight_layout()
            # plt.show()
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
                #reconstructed_catalog = fill_pixcatalog(gen_valid, reconstructed_catalog, ImageIDList, self.instr_noise)
                # _, __, instr_noise = sigma_clipped_stats(np.squeeze(Y), sigma=2.0, maxiters=50)
                # print(instr_noise*1000)
                reconstructed_catalog = fill_pixcatalog(gen_valid, reconstructed_catalog, ImageIDList, self.instr_noise)

                if batch_idx == (its - 1):
                    generated_images_model = self.generator(X, training=False).numpy()

                # evaluate_custom_source_detection(tf.squeeze(Y[0]), 8, 21)

            # Convert the generated catalog to a Pandas DataFrame
            reconstructed_catalog = pd.DataFrame(reconstructed_catalog, columns=self.cat_cols).astype(self.dtypes)
            
            # Perform Blind Source Matching
            blind_coincidence_matching_args = {"max_offset_pixels": 30, "max_distance": 7.9, "ReproductionRatio_min": 0.1, "ReproductionRatio_max": 3}
            blind_matching_args = {k: v for k, v in blind_coincidence_matching_args.items() if k != "max_offset_pixels"}

            blind_matches_catalog = find_matches(self.target_catalog, reconstructed_catalog, **blind_matching_args, return_df=True)
            # Create the Positional offset vs flux plot and determine search radius
            search_r = PS_plot(blind_matches_catalog, rnd_its, os.path.join(tdir, "PS_Plot.pdf"), (self.target_catalog, reconstructed_catalog), **blind_coincidence_matching_args)

            # Second loop for model comparison if ENABLED
            if self.model_comparison:
                # Load the correct model
                self.LoadModel(kind=kind[idx], model_path=self.model_path2)

                reconstructed_catalog2 = {"peak": [], "xpix": [], "ypix": [], "ImageID": []}
                
                for batch_idx in tqdm(range(its), desc="Super-Resolving Test Data with model2!"):
                    # Load the batch of data
                    X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                    Y = self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                    
                    # Compute the corresponding ImageIDs
                    idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
                    ImageIDList = idx_arr.tolist()

                    # Super-Resolve the batch of test data
                    gen_valid = self.generator(X, training=False).numpy()

                    # Fill the generated catalog with detected sources
                    reconstructed_catalog2 = fill_pixcatalog(gen_valid, reconstructed_catalog2, ImageIDList, self.instr_noise)

                    if batch_idx == (its - 1):
                        generated_images_model2 = self.generator(X, training=False).numpy()

                # Convert the generated catalog to a Pandas DataFrame
                reconstructed_catalog2 = pd.DataFrame(reconstructed_catalog2, columns=self.cat_cols).astype(self.dtypes)

            for ID in range(generated_images_model.shape[0]):
                Plot_InputImages(X[ID], os.path.join(tdir, f"InputHerschelImages_{ID}.pdf"))
                
                # First pick two highlighted sources/regions from the target image
                brightest, median_brightest = plot_super_resolved_image(Y[ID], "Target Image", self.target_catalog, idx_arr[ID], os.path.join(tdir, f"TargetHerschelImages_{ID}.pdf"))

                # We have not found two nice regions
                if brightest is None:
                    continue

                sources = [brightest, median_brightest]
                
                # Second project these regions on the reconstructed image
                gen_brightest, gen_median_brightest = plot_super_resolved_image(generated_images_model[ID], "Reconstructed Image", reconstructed_catalog, idx_arr[ID], os.path.join(tdir, f"ReconstructedHerschelImages_{ID}.pdf"), sources=sources)
                
                # Last image, if model_comparison is enabled
                if self.model_comparison:
                    gen2_brightest, gen2_median_brightest = plot_super_resolved_image(generated_images_model2[ID], "Model2: Reconstructed Image", reconstructed_catalog2, idx_arr[ID], os.path.join(tdir, f"Model2ReconstructedHerschelImages_{ID}.pdf"), sources=sources)
                    source_profile_comparison([Y[ID], generated_images_model[ID]], [[brightest, gen_brightest, gen2_brightest],[median_brightest, gen_median_brightest, gen2_median_brightest]], self.instr_noise, os.path.join(tdir, f"FluxProfileComparison_{ID}.pdf"))
                else:
                    source_profile_comparison([Y[ID], generated_images_model[ID]], [[brightest, gen_brightest],[median_brightest, gen_median_brightest]], self.instr_noise, os.path.join(tdir, f"FluxProfileComparison_{ID}.pdf"))


            # Using new search radius, compute the flux distribution of the matched sources
            coincidence_refined_matching_args = {"max_offset_pixels": 30, "max_distance": 7.9, "ReproductionRatio_min": 0.1, "ReproductionRatio_max": 2.5}
            refined_matching_args = {k: v for k, v in coincidence_refined_matching_args.items() if k!= "max_offset_pixels"}

            if self.model_comparison:
                refined_matches_catalog = find_matches(self.target_catalog, reconstructed_catalog2, **refined_matching_args, return_df=True)
                H2, x2_centers, y2_centers = FluxMatch_Distribution_plot(refined_matches_catalog, rnd_its, os.path.join(tdir, "2DFluxDistribution_Model2_Plot.pdf"), (self.target_catalog, reconstructed_catalog2), **coincidence_refined_matching_args)
            
                refined_matches_catalog = find_matches(self.target_catalog, reconstructed_catalog, **refined_matching_args, return_df=True)
                H, x_centers, y_centers = FluxMatch_Distribution_plot(refined_matches_catalog, rnd_its, os.path.join(tdir, "2DFluxDistribution_Plot.pdf"), (self.target_catalog, reconstructed_catalog), **coincidence_refined_matching_args)

                counts = [H, H2]

                confusion_dict1 = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
                confusion_dict2 = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}

                confusion_df1 = pd.DataFrame(confusion_dict1)
                confusion_df2 = pd.DataFrame(confusion_dict2)

                # Completeness and Reliability metrics
                confusion_df1 = confusion_score(self.target_catalog, reconstructed_catalog, confusion_df1, coincidence_refined_matching_args)
                confusion_df2 = confusion_score(self.target_catalog, reconstructed_catalog2, confusion_df2, coincidence_refined_matching_args)


                confusion_df = [confusion_df1, confusion_df2]
                label_list = ["WGANGP", "PaperModel"]
            else:
                refined_matches_catalog = find_matches(self.target_catalog, reconstructed_catalog, **refined_matching_args, return_df=True)
                H, x_centers, y_centers = FluxMatch_Distribution_plot(refined_matches_catalog, rnd_its, os.path.join(tdir, "2DFluxDistribution_Plot.pdf"), (self.target_catalog, reconstructed_catalog), **coincidence_refined_matching_args)

                counts = [H]

                confusion_dict = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
                confusion_df = pd.DataFrame(confusion_dict)

                # Completeness and Reliability metrics
                confusion_df = confusion_score(self.target_catalog, reconstructed_catalog, confusion_df, coincidence_refined_matching_args)
                confusion_df = [confusion_df]
                label_list = ["WGANGP"]

            
            FluxReproduction_plot(counts, x_centers, y_centers, self.instr_noise, os.path.join(tdir, "FluxReproduction_Plot.pdf"))
            # Make plots
            confusion_plot(self.flux_bins, confusion_df, label_list, os.path.join(tdir, "confusionscore_Plot.pdf"))
            

if __name__ == "__main__":
    SRModelTest = SRTesterGPU("TrainingConfig.ini")
    SRModelTest.LoadTestData()
    SRModelTest.TestAnalysis()