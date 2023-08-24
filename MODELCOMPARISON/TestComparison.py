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

def write_metrics_to_file(metric_val, metric_lbl, confusion_df, save):
    with open(save, "w") as f:
        f.write("Metric Values:\n")
        for lbl, val in zip(metric_lbl, metric_val):
            f.write(f"{lbl}: {val}\n")
        f.write("\nCompleteness Values\n")
        for lbl, val in zip(confusion_df['Flux bins'], confusion_df['C']):
            f.write(f"{lbl}: {val}\n")
        f.write("\nReliability Values\n")
        for lbl, val in zip(confusion_df['Flux bins'], confusion_df['R']):
            f.write(f"{lbl}: {val}\n")

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
        ## Indicate purpose of run

        self.models = [i.strip(' ') for i in self.config["MODELS"]['models'].rstrip().lstrip().split(",")]
        self.figures_output_dir = self.config['COMMON']['figures_output_dir'].strip()
        self.comparison_name = self.config['COMMON']['comparison_name'].rstrip().lstrip()
        
        ## Load path to models dir
        self.models_lib_path = self.config['COMMON']['model_outdir'].strip()
        self.models_path = [os.path.join(self.models_lib_path, model) for model in self.models]
        self.kind = [i.strip(' ') for i in self.config["MODELS"]['model_types'].rstrip().lstrip().split(",")]

        ## Set classes
        self.classes = [i.strip(' ') for i in self.config['COMMON']['input'].rstrip().lstrip().split(",")] + [self.config['COMMON']['target'].rstrip().lstrip()] # Inp first, target last

        self.TOTAL_SAMPLES = len([entry for entry in os.listdir(os.path.join(self.path_test, self.classes[0]))])

        self.tdir_out = os.path.join(self.figures_output_dir, self.comparison_name) if len(self.models) > 1 else os.path.join(self.models_path[0], self.comparison_name)

        if not os.path.isdir(self.figures_output_dir):
            os.mkdir(self.figures_output_dir)
        
        if not os.path.isdir(self.tdir_out):
            os.mkdir(self.tdir_out)
        

    def LoadTestData(self):
        self.TEST_BATCH_SIZE = 6

        self.test_arr_X = np.zeros((self.TOTAL_SAMPLES, 4, 106, 106))
        self.test_arr_Y = np.zeros((self.TOTAL_SAMPLES, 1, 424, 424))

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
        rnd_its = 25 #0 # Number of times to obtain a good estimate of distributions with randomness
        # Load correct model
        self.LoadModel(kind=self.kind[0], model_path=self.models_path[0])

        # Generated Source Catalog needs to be filled first  
        reconstructed_catalog = {"peak": [], "xpix": [], "ypix": [], "ImageID": []}
        its = self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE if self.test_arr_X.shape[0] > self.TEST_BATCH_SIZE else 1

        max_bin_value = 150 # mJy
        num_bins = 20
        flux_bin_edges = np.logspace(np.log10(1), np.log10(max_bin_value), num_bins + 1, base=10)/1000

        ## Zip the values array with itself shifted by one position to the left to create tuples of the left and right bounds of each bin
        self.flux_bins = list(zip(flux_bin_edges[:-1], flux_bin_edges[1:]))
        zero_list = np.zeros(len(self.flux_bins))

        self.PSNR = 0
        for batch_idx in tqdm(range(its), desc="Super-Resolving Test Data with main model!"):
            # Load the batch of data
            X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            Y = self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            
            # Compute the corresponding ImageIDs
            idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
            ImageIDList = idx_arr.tolist()

            # Super-Resolve the batch of test data
            gen_valid = self.generator(X, training=False).numpy()

            self.PSNR += compute_PSNR_batch(gen_valid, Y)/its

            # Fill the generated catalog with detected sources
            reconstructed_catalog = fill_pixcatalog(gen_valid, reconstructed_catalog, ImageIDList, self.instr_noise)

            # if batch_idx == (its - 1):
            #     generated_images_model = self.generator(X, training=False).numpy()

            # evaluate_custom_source_detection(tf.squeeze(Y[0]), 8, 21)

        # Convert the generated catalog to a Pandas DataFrame
        reconstructed_catalog = pd.DataFrame(reconstructed_catalog, columns=self.cat_cols).astype(self.dtypes)
        
        generated_images_IDs = np.array([0, 1, 50, 51, self.test_arr_Y.shape[0]-2, self.test_arr_Y.shape[0]-1], dtype=np.int32)
        generated_images_model1 = self.generator(self.test_arr_X[generated_images_IDs], training=False).numpy()

        # Perform Blind Source Matching
        blind_coincidence_matching_args = {"max_offset_pixels": 30, "max_distance": 7.9, "ReproductionRatio_min": 0.1, "ReproductionRatio_max": 3}
        blind_matching_args = {k: v for k, v in blind_coincidence_matching_args.items() if k != "max_offset_pixels"}

        blind_matches_catalog1 = find_matches(self.target_catalog, reconstructed_catalog, **blind_matching_args, return_df=True)
        # Create the Positional offset vs flux plot and determine search radius
        search_r = PS_plot(blind_matches_catalog1, rnd_its, os.path.join(self.tdir_out, f"{self.models[0]}_PS_Plot.png"), (self.target_catalog, reconstructed_catalog), **blind_coincidence_matching_args)

        # Second loop for model comparison if ENABLED
        if len(self.models) > 1:
            # Load the correct model
            self.LoadModel(kind=self.kind[1], model_path=self.models_path[1])

            reconstructed_catalog2 = {"peak": [], "xpix": [], "ypix": [], "ImageID": []}
            self.PSNR2 = 0
            for batch_idx in tqdm(range(its), desc="Super-Resolving Test Data with model2!"):
                # Load the batch of data
                X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                Y = self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                
                # Compute the corresponding ImageIDs
                idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
                ImageIDList = idx_arr.tolist()

                # Super-Resolve the batch of test data
                gen_valid = self.generator(X, training=False).numpy()

                self.PSNR2 += compute_PSNR_batch(gen_valid, Y)/its
                # Fill the generated catalog with detected sources
                reconstructed_catalog2 = fill_pixcatalog(gen_valid, reconstructed_catalog2, ImageIDList, self.instr_noise)

                # if batch_idx == (its - 1):
                #     generated_images_model2 = self.generator(X, training=False).numpy()

            # Convert the generated catalog to a Pandas DataFrame
            reconstructed_catalog2 = pd.DataFrame(reconstructed_catalog2, columns=self.cat_cols).astype(self.dtypes)
            blind_matches_catalog2 = find_matches(self.target_catalog, reconstructed_catalog2, **blind_matching_args, return_df=True)

            search_r = PS_plot(blind_matches_catalog2, rnd_its, os.path.join(self.tdir_out, f"{self.models[1]}_PS_Plot.png"), (self.target_catalog, reconstructed_catalog), **blind_coincidence_matching_args)
            generated_images_model2 = self.generator(self.test_arr_X[generated_images_IDs], training=False).numpy()

        for i, ID in enumerate(generated_images_IDs):
            Plot_InputImages(self.test_arr_X[ID], os.path.join(self.tdir_out, f"InputHerschelImages_{ID}.png"))
            
            # First pick two highlighted sources/regions from the target image
            brightest, median_brightest = plot_super_resolved_image(self.test_arr_Y[ID], "Target Image", self.target_catalog, ID, os.path.join(self.tdir_out, f"TargetHerschelImages_{ID}.png"))

            # We have not found two nice regions
            if brightest is None:
                continue

            sources = [brightest, median_brightest]
            
            # Second project these regions on the reconstructed image
            gen_brightest, gen_median_brightest = plot_super_resolved_image(generated_images_model1[i], f"{self.models[0]}: Super-resolved Image", reconstructed_catalog, ID, os.path.join(self.tdir_out, f"{self.models[0]}_ReconstructedHerschelImages_{ID}.png"), sources=sources)
            
            # Last image, if model_comparison is enabled
            if len(self.models) > 1:
                gen2_brightest, gen2_median_brightest = plot_super_resolved_image(generated_images_model2[i], f"{self.models[1]}: Super-resolved Image", reconstructed_catalog2, ID, os.path.join(self.tdir_out, f"{self.models[1]}_ReconstructedHerschelImages_{ID}.png"), sources=sources)
                source_profile_comparison([self.test_arr_Y[ID], generated_images_model1[i], generated_images_model2[i]], [[brightest, gen_brightest, gen2_brightest],[median_brightest, gen_median_brightest, gen2_median_brightest]], self.instr_noise, ["Horizontal True Profile", f"{self.models[0]}: Horizontal Generated Profile", f"{self.models[1]}: Horizontal Generated Profile"], os.path.join(self.tdir_out, f"FluxProfileComparison_{ID}.png"))
            else:
                source_profile_comparison([self.test_arr_Y[ID], generated_images_model1[i]], [[brightest, gen_brightest],[median_brightest, gen_median_brightest]], self.instr_noise, ["Horizontal True Profile", "Horizontal Generated Profile"], os.path.join(self.tdir_out, f"FluxProfileComparison_{ID}.png"))


        # compute the flux distribution of the matched sources
        if len(self.models) > 1:  
            H2, x2_centers, y2_centers, mape2 = FluxMatch_Distribution_plot(blind_matches_catalog2, rnd_its, os.path.join(self.tdir_out, f"2DFluxDistribution_{self.models[1]}_Plot.png"), (self.target_catalog, reconstructed_catalog2), **blind_coincidence_matching_args)
        
            H, x_centers, y_centers, mape = FluxMatch_Distribution_plot(blind_matches_catalog1, rnd_its, os.path.join(self.tdir_out, f"2DFluxDistribution_{self.models[0]}_Plot.png"), (self.target_catalog, reconstructed_catalog), **blind_coincidence_matching_args)

            counts = [H, H2]

            confusion_dict1 = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
            confusion_dict2 = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}

            confusion_df1 = pd.DataFrame(confusion_dict1)
            confusion_df2 = pd.DataFrame(confusion_dict2)

            # Completeness and Reliability metrics
            confusion_df1 = confusion_score(self.target_catalog, reconstructed_catalog, confusion_df1, rnd_its, blind_coincidence_matching_args)
            confusion_df2 = confusion_score(self.target_catalog, reconstructed_catalog2, confusion_df2, rnd_its, blind_coincidence_matching_args)

            confusion_df = [confusion_df1, confusion_df2]
            # Global completeness and reliability metrics
            Cglob1 = np.sum(confusion_df1["TPc"])/(np.sum(confusion_df1["TPc"]) + np.sum(confusion_df1["FNc"]))
            Rglob1 = np.sum(confusion_df1["TPr"])/(np.sum(confusion_df1["TPr"]) + np.sum(confusion_df1["FPr"]))

            Cglob2 = np.sum(confusion_df2["TPc"])/(np.sum(confusion_df2["TPc"]) + np.sum(confusion_df2["FNc"]))
            Rglob2 = np.sum(confusion_df2["TPr"])/(np.sum(confusion_df2["TPr"]) + np.sum(confusion_df2["FPr"]))

            write_metrics_to_file([self.PSNR, Cglob1, Rglob1, mape*100], ["PSNR", "C Global", "R Global", "MAPE"], confusion_df1, os.path.join(self.tdir_out, f"TestMetricResults_{self.models[0]}.txt"))
            write_metrics_to_file([self.PSNR2, Cglob2, Rglob2, mape2*100], ["PSNR", "C Global", "R Global", "MAPE"], confusion_df2, os.path.join(self.tdir_out, f"TestMetricResults_{self.models[1]}.txt"))

        else:
            H, x_centers, y_centers, mape = FluxMatch_Distribution_plot(blind_matches_catalog1, rnd_its, os.path.join(self.tdir_out, "2DFluxDistribution_Plot.png"), (self.target_catalog, reconstructed_catalog), **blind_coincidence_matching_args)

            counts = [H]

            confusion_dict = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
            confusion_df = pd.DataFrame(confusion_dict)

            # Completeness and Reliability metrics
            confusion_df = confusion_score(self.target_catalog, reconstructed_catalog, confusion_df, rnd_its, blind_coincidence_matching_args)
            confusion_df = [confusion_df]

            write_metrics_to_file([self.PSNR], ["PSNR"], confusion_df, os.path.join(self.tdir_out, f"TestMetricResults_{self.models[0]}.txt"))
        print("confusion_b4_plot", confusion_df[0])
        FluxReproduction_plot(counts, x_centers, y_centers, self.instr_noise, self.models, os.path.join(self.tdir_out, "FluxReproduction_Plot.png"))
        # Make plots
        confusion_plot(self.flux_bins, confusion_df, self.models, os.path.join(self.tdir_out, "confusionscore_Plot.png"))
            

if __name__ == "__main__":
    SRModelTest = SRTesterGPU("TestComparison.ini")
    SRModelTest.LoadTestData()
    SRModelTest.TestAnalysis()