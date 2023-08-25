######################
###     IMPORTS    ###
######################

import tensorflow as tf
tf.random.set_seed(42)

import os
import configparser
import gc
import time

import numpy as np
import pandas as pd

from astropy.io import fits
from PlotLib.PlotFunctionsTest import *
from metric_funcs import *
from astropy.wcs import WCS
from tqdm import tqdm

from astroML import correlation as corr
#######################
###    TEST CLASS   ###
#######################

class SRTesterGPU():
    def __init__(self, path_to_config, correlation_data_path, gridmode = False, idx = None) -> None:
        # Load configuration file for training
        self.config = configparser.ConfigParser()
        self.config.read(path_to_config)

        # Create any missing directories and initialize necessary parameters
        self.DIM = (424, 424)

        ## Paths with train data
        self.path_test = self.config['COMMON']['path_test'].rstrip().lstrip()
        self.correlation_data_path = correlation_data_path 
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
        self.TOTAL_SAMPLES_CORR = len([entry for entry in os.listdir(os.path.join(self.correlation_data_path, self.classes[0]))])
        self.tdir_out = [os.path.join(self.model_path, "test_results_BestConfusion"), os.path.join(self.model_path, "test_results_BestValid"), os.path.join(self.model_path, "test_results_BestFluxReproduction")]

        for tdir in self.tdir_out:
            if not os.path.isdir(tdir):
                os.mkdir(tdir)

        #Register GridMode
        self.gridmode = gridmode

    def LoadTestData(self):
        self.TEST_BATCH_SIZE = 24

        self.test_arr_X = np.zeros((self.TOTAL_SAMPLES, 3, 106, 106))
        self.test_arr_Y = np.zeros((self.TOTAL_SAMPLES, 1, 424, 424))

        self.test_arr_X_corr = np.zeros((self.TOTAL_SAMPLES_CORR, 3, 106, 106))
        self.test_arr_Y_corr = np.zeros((self.TOTAL_SAMPLES_CORR, 1, 424, 424))

        self.test_wcs_corr = []

        for i in tqdm(range(self.TOTAL_SAMPLES), desc=f"Loading Data From {self.path_test}"):
            for k in range(len(self.classes)):
                with fits.open(os.path.join(self.path_test, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
                    if k == len(self.classes) - 1:
                        self.test_arr_Y[i] = hdu[0].data
                        arr = np.array([list(row) for row in hdu[1].data])
                        if i == 0:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = arr.copy()
                        else:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = np.vstack((self.target_image_sources_cat_test, arr))

                        del arr;
                    else:
                        self.test_arr_X[i][k] = hdu[0].data   

        for i in tqdm(range(self.TOTAL_SAMPLES_CORR), desc=f"Loading Data From {self.correlation_data_path}"):
            for k in range(len(self.classes)):
                with fits.open(os.path.join(self.correlation_data_path, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
                    if k == len(self.classes) - 1:
                        self.test_arr_Y_corr[i] = hdu[0].data
                        self.test_wcs_corr.append(WCS(hdu[0].header))
                    else:
                        self.test_arr_X_corr[i][k] = hdu[0].data  

        # Free memory
        gc.collect()


    def LoadModel(self, kind):
        self.generator = tf.keras.models.load_model(os.path.join(self.model_path, f'{kind}_Model'))
        gc.collect()


    def TestAnalysis(self, α = None):
        # Check α
        if self.gridmode == True:
            assert α is not None, "α must be given"
        else:
            α = float(self.config['TRAINING PARAMETERS']['alpha'].rstrip().lstrip())

        kind = ["BestConfusion", "BestValid", "BestFluxReproduction"]
        inset_its = 4
        rnd_its = 10 # Number of times to obtain a good estimate of distributions with randomness
        nbins_relSdiff = 30
        bin_edges_relSpeakdiff = np.logspace(np.log10(10), np.log10(125), nbins_relSdiff+1)
        bin_edges_relSaperdiff = np.logspace(np.log10(1), np.log10(50), nbins_relSdiff+1)
        for idx, tdir in tqdm(enumerate(self.tdir_out), desc="Testing Models"):
            # Load correct model
            self.LoadModel(kind=kind[idx])

            # Initialisation for reproduction plot
            self.mean_relSpeakdiff_arr = np.zeros((rnd_its, nbins_relSdiff))
            self.MAD_relSpeakdiff_arr = np.zeros((rnd_its, nbins_relSdiff))

            self.mean_relSaperdiff_arr = np.zeros((rnd_its, nbins_relSdiff))
            self.MAD_relSaperdiff_arr = np.zeros((rnd_its, nbins_relSdiff))


            # Calculate Metrics
            self.Lh = 0; self.Lstats = 0; self.Lflux = 0; self.Laperflux = 0; self.Lpeakflux = 0; self.Lssim = 0;
            ## Create an array of logarithmically spaced values from min to max bin value
            max_bin_value = 150 # mJy
            num_bins = 30
            flux_bin_edges = np.logspace(np.log10(10), np.log10(max_bin_value), num_bins + 1, base=10)/1000
            bin_edges_corr = np.linspace(4/3600, 1500/3600, 50)

            ## Zip the values array with itself shifted by one position to the left to create tuples of the left and right bounds of each bin
            self.flux_bins = list(zip(flux_bin_edges[:-1], flux_bin_edges[1:]))
            zero_list = np.zeros(len(self.flux_bins))
            confusion_dict = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
            confusion_df = pd.DataFrame(confusion_dict)

            PSNR = 0

            matches_df_arr_peak = [{'Sout1':[], 'Sout2':[], "Sout3":[], "Sin_true":[], "Sin_rnd":[],"Sin_flag":[], "true_offset":[], "rnd_offset":[], "flag_offset":[], "true_xy_offset":[[],[]], "rnd_xy_offset":[[],[]], "flag_xy_offset":[[],[]]} for i in range(rnd_its)]
            matches_df_arr_aper = [{'Sout1':[], 'Sout2':[], "Sout3":[], "Sin_true":[], "Sin_rnd":[],"Sin_flag":[], "true_offset":[], "rnd_offset":[], "flag_offset":[], "true_xy_offset":[[],[]], "rnd_xy_offset":[[],[]], "flag_xy_offset":[[],[]]} for i in range(rnd_its)]

            its = self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE
            for batch_idx in tqdm(range(its), desc="Evaluating on Test Data"):
                X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                Y = self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                Y_source_cat = list()
                idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
                for j, k in enumerate(idx_arr):
                    Y_source_cat.append(self.target_image_sources_cat_test[np.where(self.target_image_sources_cat_test[:,-1] == k)])
                    # Synchronize batch idx and catalogue image idx
                    Y_source_cat[-1][:,-1] = j
                Y_source_cat = np.vstack(Y_source_cat)
                gen_valid = self.generator(X, training=False).numpy()

                # Loss metrics
                Lh_batch, Lstats_batch, Laperflux_batch, Lpeakflux_batch, Lssim = loss_test(gen_valid, Y, Y_source_cat, α)
                self.Lh += Lh_batch; self.Lstats += Lstats_batch; self.Laperflux += Laperflux_batch; self.Lpeakflux += Lpeakflux_batch; self.Lssim += Lssim/(its)


                # PSNR
                PSNR += compute_PSNR_batch(gen_valid, Y)/its

                # Completeness and Reliability metrics
                confusion_df = confusion_score(gen_valid, Y, Y_source_cat, confusion_df)

                # Hexplot metrics
                for i in range(rnd_its):
                    matches_df_arr_peak[i], matches_df_arr_aper[i]  = find_matches(gen_valid, Y, Y_source_cat, matches_df_arr_peak[i], matches_df_arr_aper[i])
            
                # Insetplot
                # Tweak the number of loops to produce the number of desired plots, note that this number does not equal number of plots!
                # additional plot spam filter
                if batch_idx < 25:
                    for i in range(inset_its):
                        insetplot(X[i], gen_valid[i], Y[i], Y_source_cat[np.where(Y_source_cat[:,-1] == i)], os.path.join(tdir, f"insetplot_{batch_idx}_{i}.pdf"))
                        #insetplot_diff(X[i], gen_valid[i], Y[i], Y_source_cat[np.where(Y_source_cat[:,-1] == i)], os.path.join(tdir, f"insetplotdiff_{batch_idx}_{i}.pdf"))

                        if i == 0:
                            plot_kernelprofile(Y[i], gen_valid[i], 21, os.path.join(tdir, f"KernelProfile_{batch_idx}_{i}.pdf"))

            # Calculate completeness and reliability of test sample
            ## If needed resolve zero-occurences
            for i in range(len(confusion_df['Flux bins'])):
                if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
                    confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])
            
                if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
                    confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
                    confusion_df.loc[i, 'flag_R'] = confusion_df.loc[i, 'flag_TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
        

            # Set average completeness, reliability
            self.avg_c = np.mean(confusion_df['C'])
            self.avg_r = np.mean(confusion_df['R'])


            # Compute relative flux differences, mean and spread
            for it in range(rnd_its):
                mean, MAD = FluxReproduction(matches_df_arr_peak[it], bin_edges_relSpeakdiff)
                self.mean_relSpeakdiff_arr[it] = mean
                self.MAD_relSpeakdiff_arr[it] = MAD
                mean, MAD = FluxReproduction(matches_df_arr_aper[it], bin_edges_relSaperdiff)
                self.mean_relSaperdiff_arr[it] = mean
                self.MAD_relSaperdiff_arr[it] = MAD

            mean_relSpeakdiff = np.mean(self.mean_relSpeakdiff_arr, axis=0)
            mean_err_relSpeakdiff = np.std(self.mean_relSpeakdiff_arr, axis=0)
            mean_relSaperdiff = np.mean(self.mean_relSaperdiff_arr, axis=0)
            mean_err_relSaperdiff = np.std(self.mean_relSaperdiff_arr, axis=0)
            MAD_relSpeakdiff = np.mean(self.MAD_relSpeakdiff_arr, axis=0)
            MAD_err_relSpeakdiff = np.std(self.MAD_relSpeakdiff_arr, axis=0)
            MAD_relSaperdiff = np.mean(self.MAD_relSaperdiff_arr, axis=0)
            MAD_err_relSaperdiff = np.std(self.MAD_relSaperdiff_arr, axis=0)
            # Calculate the angular two point correlation function
            ## Angular Two-Point Correlation function loop 
            Y_catalog_target = {'RA':[], 'DEC':[], 'S':[]}
            Y_catalog_gen = {'RA':[], 'DEC':[], 'S':[]}

            # Angular correlation function
            its = self.test_arr_Y_corr.shape[0]//self.TEST_BATCH_SIZE
            for batch_idx in tqdm(range(its), desc="Evaluating on Clustered Data"):
                X = self.test_arr_X_corr[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X_corr[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                Y = self.test_arr_Y_corr[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_Y_corr[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
                gen_valid = self.generator(X, training=False).numpy()

                wcs_arr = self.test_wcs_corr[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)] if batch_idx != (its - 1) else self.test_wcs_corr[batch_idx*self.TEST_BATCH_SIZE:]


                Y_catalog_target = fill_catalog(Y, wcs_arr, Y_catalog_target)
                Y_catalog_gen = fill_catalog(gen_valid, wcs_arr, Y_catalog_gen)
            ## Calculate the correlation function for target once
            mean_target, std_target, _ = corr.bootstrap_two_point_angular(ra=Y_catalog_target['RA'], dec=Y_catalog_target['DEC'], bins=bin_edges_corr, method='landy-szalay')
            self.corr_target = mean_target
            self.corr_err_target = std_target

            ## The angular correlation function for the generated sample
            mean_gen, std_gen, _ = corr.bootstrap_two_point_angular(ra=Y_catalog_gen['RA'], dec=Y_catalog_gen['DEC'], bins=bin_edges_corr, method='landy-szalay')

            ## Random Baseline check to confirm correctness of Angular two point function
            ra_rnd = np.random.uniform(np.min(Y_catalog_target['RA']), np.max(Y_catalog_target['RA']), size=len(Y_catalog_target['RA']))
            dec_rnd = np.random.uniform(np.min(Y_catalog_target['DEC']), np.max(Y_catalog_target['DEC']), size=len(Y_catalog_target['RA']))
            mean_rnd, std_rnd, _ = corr.bootstrap_two_point_angular(ra=ra_rnd, dec=dec_rnd, bins=bin_edges_corr, method='landy-szalay')


            # Make plots
            confusion_plot(self.flux_bins, confusion_df, os.path.join(tdir, "confusionplot.pdf"))
            PS_plot(matches_df_arr_peak, 'peak', os.path.join(tdir, "PSpeak_plot.png"))
            PS_plot(matches_df_arr_aper, 'aper', os.path.join(tdir, "PSaper_plot.png"))
            hexplot(matches_df_arr_peak, 'peak', os.path.join(tdir, "2DHistMatchesSpeak_plot.png"))
            hexplot(matches_df_arr_aper, 'aper', os.path.join(tdir, "2DHistMatchesSaper_plot.png"))
            FluxReproduction_plot((bin_edges_relSpeakdiff[:-1] + bin_edges_relSpeakdiff[1:])/2, mean_relSpeakdiff, mean_err_relSpeakdiff, MAD_relSpeakdiff, MAD_err_relSpeakdiff, 'peak', os.path.join(tdir, "PeakFluxReproduction_plot.pdf"))
            FluxReproduction_plot((bin_edges_relSaperdiff[:-1] + bin_edges_relSaperdiff[1:])/2, mean_relSaperdiff, mean_err_relSaperdiff, MAD_relSaperdiff, MAD_err_relSaperdiff, 'aper', os.path.join(tdir, "AperFluxReproduction_plot.pdf"))
            CorrelationPlotCheck(self.corr_target, self.corr_err_target, mean_rnd, std_rnd, (bin_edges_corr[:-1] + bin_edges_corr[1:])/2, os.path.join(tdir, "CorrelationCheck_plot.pdf"))
            CorrelationPlot(self.corr_target, self.corr_err_target, mean_gen, std_gen, (bin_edges_corr[:-1] + bin_edges_corr[1:])/2, os.path.join(tdir, "Correlation_plot.pdf"))

if __name__ == "__main__":
    SRModelTest = SRTesterGPU("TrainingConfig.ini", r"/home1/s3101940/HERSPIRESRproj/simData/SIDES_clustered_noaugment")
    SRModelTest.LoadTestData()
    SRModelTest.TestAnalysis(α = 0.01)