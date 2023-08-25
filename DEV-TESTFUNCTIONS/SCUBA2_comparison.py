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
        self.path_test = self.config['COMMON']['path_scuba2_test'].rstrip().lstrip()
        # self.correlation_data_path = correlation_data_path 
        ## Indicate purpose of run
        self.RUN_NAME = self.config['MODELS']['model_name'].rstrip().lstrip()

        ## Load path to models dir
        self.models_lib_path = self.config['COMMON']['model_outdir'].rstrip().lstrip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)
        self.kind = self.config['MODELS']['model_type'].rstrip().lstrip()

        ## Paths to SCUBA2 catalog
        self.path_scuba_cat = self.config['COMMON']['path_scuba2_catalog'].rstrip().lstrip()

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
                    else:
                        self.test_arr_X[i][k] = hdu[0].data   

        self.dtypes = {"peak": np.float32, "xpix": np.float32, "ypix":np.float32, "ra":np.float32, "dec":np.float32, "ImageID":np.int32}
        self.cat_cols = ["peak", "xpix", "ypix", "ra", "dec", "ImageID"]
        # Load in the Simulation master catalog
        self.target_cat = pd.DataFrame(data=self.target_image_sources_cat_test, columns=self.cat_cols).astype(self.dtypes)
        self.scuba_cat = pd.DataFrame(fits.getdata(self.path_scuba_cat))
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


    def match_with_scuba_catalog(self, scuba_catalog, reconstructed_catalog, max_distance=4, ReproductionRatio_min=0.5, ReproductionRatio_max=1.5, return_df=False):
        matches_catalog = {"Scuba-2 Catalog Source Flux": [], "Scuba-2 Catalog Source Flux err": [], "Detected Source Flux": [], "Distance" : []}
        ## Iterate over target catalog
        ## Find the best match
        ## Fill the matches catalog
        ## Note that we work with mJy in this script with SCUBA2
        for detected_source_idx, detected_source in reconstructed_catalog.iterrows():
            r = np.sqrt((detected_source['ra'] - scuba_catalog['RA'])**2 + (detected_source['dec'] - scuba_catalog['Dec'])**2)
            rmin_idx = np.argmin(r)
            if r.values[rmin_idx]*3600 <= max_distance: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                matches_catalog["Detected Source Flux"].append(detected_source['peak']*1000)
                matches_catalog["Scuba-2 Catalog Source Flux"].append(scuba_catalog['S450'].values[rmin_idx])
                matches_catalog["Scuba-2 Catalog Source Flux err"].append(scuba_catalog['S450_total_err'].values[rmin_idx])
                matches_catalog["Distance"].append(r.values[rmin_idx]*3600) # In arcseconds

        if return_df:
            cols = ["Scuba-2 Catalog Source Flux", "Scuba-2 Catalog Source Flux err", "Detected Source Flux", "Distance"]
            return pd.DataFrame(matches_catalog, columns=cols)
        else:
            return matches_catalog
    

    def coincidence_matches_world(self, Reconstructed_catalog, return_df=False, **matching_args):
        scuba_catalog_copy = self.scuba_cat.copy()
        # Mutate the target catalog x and y coordinates by a random offset
        scuba_catalog_copy[["RA", "Dec"]] += np.random.uniform(low=-30/3600, high=30/3600, size=scuba_catalog_copy[["RA", "Dec"]].shape)
        
        # Perform random Blind Source Matching  
        # The resulting catalog contains fake matches
        rnd_matches_catalog = self.match_with_scuba_catalog(scuba_catalog_copy, Reconstructed_catalog, **matching_args, return_df=return_df)
        if return_df:
            cols = ["Scuba-2 Catalog Source Flux", "Scuba-2 Catalog Source Flux err", "Detected Source Flux", "Distance"]
            return pd.DataFrame(rnd_matches_catalog, columns=cols)
        else:
            return rnd_matches_catalog

    def plot_source_recovery(self, cat_matches, ylabel, save_path):
        plt.figure(figsize=(7,7))

        xmax = np.max(cat_matches["Scuba-2 Catalog Source Flux"])
        true_line = np.linspace(0, xmax+5, 100)

        plt.errorbar(cat_matches["Scuba-2 Catalog Source Flux"], cat_matches["Detected Source Flux"], xerr=cat_matches["Scuba-2 Catalog Source Flux err"], markersize=10, fmt='.', color='blue', label="Matched Sources", alpha=0.8)
        plt.plot(true_line, true_line, color='red', label="1:1 Recovery", linestyle='dotted', linewidth=2)
        plt.xlabel(r"SCUBA-2 $450 \mu m$ Source Flux $S_{in}$ [mJy]", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.yscale('log')
        plt.xscale('log')

        plt.xlim([1, xmax+5])
        plt.ylim([1, xmax+5])

        plt.legend(fontsize=12)
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        plt.close()

    def PS_scuba_plot(self, blind_matches_catalog, rnd_its, save, reproduction_catalog, **matching_args):
        cols = blind_matches_catalog.columns.tolist() if isinstance(blind_matches_catalog, pd.DataFrame) else blind_matches_catalog.keys()
        # Define the colormap for the 2D distribution plot
        cmap= plt.get_cmap("plasma")
        cmap.set_bad(color='black')

        # Create the corner plot
        fig = plt.figure(figsize=(7.5, 7.5))
        grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.3, width_ratios=[1,5,5,5,5])
        ax_main = fig.add_subplot(grid[1:4, 1:4])
        ax_xhist = fig.add_subplot(grid[0, 1:4], sharex=ax_main)
        ax_yhist = fig.add_subplot(grid[1:4, 4], sharey=ax_main)
        ax_colorbar = fig.add_subplot(grid[1:4, 0])
        # Create the 2D distribution plot
        binsx = 10
        binsy = 20
        xbins = np.linspace(matching_args['ReproductionRatio_min'], matching_args['ReproductionRatio_max'], binsx + 1)
        # xbins = np.linspace(-25/1000, 25/1000, binsx + 1)

        ybins = np.linspace(0, matching_args['max_distance'], binsy + 1)
        H_blind, xedges_blind, yedges_blind = np.histogram2d(blind_matches_catalog[cols[2]]/blind_matches_catalog[cols[0]], blind_matches_catalog[cols[3]], bins=(xbins, ybins))
        # H_blind, xedges, yedges = np.histogram2d(blind_matches_catalog[cols[0]] - blind_matches_catalog[cols[1]], blind_matches_catalog[cols[2]], bins=(xbins, ybins))
        #H = np.ma.masked_where(H == 0, H)
        x_centers = (xbins[:-1] + xbins[1:]) / 2
        y_centers = (ybins[:-1] + ybins[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        # Compute the fake 2d distribution
        H_rnd = np.zeros((binsx, binsy)) # H has x=rows and y=columns, H.T is the transpose
        for i in tqdm(range(rnd_its), desc="Computing the fake 2d distribution..."):
            rnd_matches_catalog = self.coincidence_matches_world(reproduction_catalog, return_df=True, **matching_args)
            H_it, xedges, yedges = np.histogram2d(rnd_matches_catalog[cols[2]]/rnd_matches_catalog[cols[0]], rnd_matches_catalog[cols[3]], bins=(xbins, ybins))
            # H_it, xedges, yedges = np.histogram2d(rnd_matches_catalog[cols[0]] - rnd_matches_catalog[cols[1]], rnd_matches_catalog[cols[2]], bins=(xbins, ybins))

            H_rnd += H_it/rnd_its

        # Compute the 2D distribution plot of good matches
        H = np.round(H_blind - H_rnd, 0).astype(np.int32)
        H[H<= 0] = 0
        mask = np.ma.masked_where(H.T == 0, H.T)
        pcol = ax_main.pcolormesh(xedges_blind, yedges_blind, mask, cmap=cmap, vmin=0)#, shading='gouraud')
        cb = fig.colorbar(pcol, ax=ax_main, cax=ax_colorbar, location='left')
        pos_cb = ax_colorbar.get_position()
        pos_main = ax_main.get_position()
        ax_colorbar.set_position([pos_cb.x0 - 0.05, pos_main.y0, 0.03, pos_main.y1 - pos_main.y0])
        cb.set_label('Number of Matches')

        # Set the axis limits for the 2D distribution plot
        ax_main.set_xlim(matching_args['ReproductionRatio_min'], matching_args['ReproductionRatio_max'])
        # ax_main.set_xlim(-25/1000, 25/1000)

        ax_main.set_ylim(0, matching_args['max_distance'])

        # Create the x and y marginal plots
        # counts1, edges1 = ax_xhist.hist(x, bins=(xedges), density=True, histtype='step', color='blue', lw=2)
        # ax_yhist.hist(y, bins=(yedges), density=True, histtype='step', color='blue', lw=2, orientation='horizontal')
        Xhist = np.sum(H.T, axis=0)/np.sum(H.T)
        Yhist = np.sum(H.T, axis=1)/np.sum(H.T)

        # Add the cumulative probability lines to the marginal plots
        #x_sorted = np.sort(x)
        #y_sorted = np.sort(y)
        cumulative_prob_reproduction_ratio = np.cumsum(Xhist)
        cumulative_prob_offset = np.cumsum(Yhist)

        index_search_r = np.searchsorted(cumulative_prob_offset, 0.95)

        ax_xhist_y2 = ax_xhist.twinx();
        ax_yhist_y2 = ax_yhist.twiny();

        ax_xhist_y2.plot(x_centers, cumulative_prob_reproduction_ratio, color='red', linestyle='--', lw=1)
        ax_xhist.plot(x_centers, Xhist, color='blue', linestyle='-', lw=1)
        ax_yhist_y2.plot(cumulative_prob_offset, y_centers, color='red', linestyle='--', lw=1)
        ax_yhist.plot(Yhist, y_centers, color='blue', linestyle='-', lw=1)

        # # Plot Truth Line
        # ax_main.plot(x_centers, y_centers, color='black', linestyle='--', lw=1)

        # Set xticks
        ax_xhist.tick_params(axis='y', which='both', labelsize=8, colors='blue')
        ax_xhist.tick_params(axis='x', which='both', labelsize=8)
        ax_xhist_y2.tick_params(axis='y', which='both', labelsize=8, colors='red')
        ax_xhist_y2.yaxis.label.set_color('blue')
        ax_yhist_y2.xaxis.label.set_color('red')
        ax_yhist.yaxis.label.set_color('blue')
        ax_yhist.yaxis.label.set_color('red')

        ax_yhist.tick_params(axis='y', which='both', labelsize=8)
        ax_yhist.tick_params(axis='x', which='both', labelsize=8, colors='blue', rotation=-90)
        ax_yhist_y2.tick_params(axis='x', which='both', labelsize=8, colors='red', rotation=-90)

        ax_main.tick_params(axis='y', which='both', labelsize=8)
        ax_main.tick_params(axis='x', which='both', labelsize=8)
        ax_main.tick_params(axis='y', which='both', labelsize=8)
        # Set the axis labels
        ax_main.set_xlabel(r'Ratio $\frac{Recovered \ Source \ Flux}{Target \ Source \ Flux}}$', fontsize=10)
        ax_main.set_ylabel('Offset (arcseconds) (\'\')', fontsize=10)
        ax_xhist.set_ylabel('PDF', fontsize=10, color='blue', rotation=-90, labelpad=15)
        ax_yhist.set_xlabel('PDF', fontsize=10, color='blue')
        ax_xhist_y2.set_ylabel('CDF', fontsize=10, color='red', rotation=-90, labelpad=10)
        ax_yhist_y2.set_xlabel('CDF', fontsize=10, color='red')

        ax_yhist.grid(which='both', alpha=0.4, color='lightgrey', linestyle='--')
        ax_xhist.grid(which='both', alpha=0.4, color='lightgrey', linestyle='--')
        # Plot vertical lines for 95% confidence interval
        xticks_xhist = np.round(np.arange(0., np.max(Xhist), np.max(Xhist)/4), 2)
        xticks_yhist = np.round(np.arange(0., np.max(Yhist), np.max(Yhist)/4), 2)
        ax_xhist.set_yticks(xticks_xhist)
        ax_yhist.set_xticks(xticks_yhist)
        ax_xhist_y2.set_yticks(np.arange(0, 1.2, 0.2))
        ax_yhist_y2.set_xticks(np.arange(0, 1.2, 0.2))

        ax_yhist_y2_xlim = ax_yhist_y2.get_xlim()
        ax_yhist_y2.hlines(xmin=ax_yhist_y2_xlim[0], xmax=ax_yhist_y2_xlim[1], y=y_centers[index_search_r], linestyle='dotted', color='black', lw=1)
        ax_yhist_y2.vlines(ymin=0, ymax=ax_yhist_y2.get_ylim()[1], x=0.95, linestyle='dotted', color='black', lw=1, label=f'95% Confidence: Offset={y_centers[index_search_r]} (\'\')')
        
        ax_yhist_y2.text(ax_yhist_y2.get_xlim()[1] + 0.25, y_centers[index_search_r], f'{y_centers[index_search_r]}(\'\')', fontsize=8, rotation=-90, ha='center', va='center')
        
        ax_yhist_y2.hlines(xmin=ax_yhist_y2_xlim[0], xmax=ax_yhist_y2_xlim[1], y=7.9, linestyle='dotted', color='black', lw=1)

        ax_yhist_y2.text(ax_yhist_y2.get_xlim()[1] + 0.1, 7.9, f'FWHM: {7.9}(\'\')', fontsize=8, rotation=-90, ha='center', va='center')

        ax_yhist_y2.hlines(xmin=ax_yhist_y2_xlim[0], xmax=ax_yhist_y2_xlim[1], y=4., linestyle='dotted', color='black', lw=1)
        
        ax_yhist_y2.text(ax_yhist_y2.get_xlim()[1] + 0.1, 4., r'$\sigma_{Input}$'f': {4.0}(\'\')', fontsize=8, rotation=-90, ha='center', va='center')
        lines_yhist_y2, labels_yhist_y2 = ax_yhist_y2.get_legend_handles_labels()
        fig.legend(lines_yhist_y2, labels_yhist_y2, loc='upper right', fontsize=10, ncol=1)
        # Display the plot
        fig.savefig(save, dpi=400, bbox_inches='tight')
        plt.close(fig)

        return np.round(y_centers[index_search_r] * 2)/2

    def Analysis(self):
        self.LoadModel(kind=self.kind, model_path=self.model_path)
        # Generated Source Catalog needs to be filled first  
        reconstructed_catalog = {"peak": [], "xpix": [], "ypix": [], "ra":[], "dec":[], "ImageID": []}
        its = self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE if self.test_arr_X.shape[0] > self.TEST_BATCH_SIZE else 1

        rnd_its = 10
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

            if batch_idx == (its - 1):
                generated_images_model = self.generator(X, training=False).numpy()
        # Convert the generated catalog to a Pandas DataFrame
        reconstructed_catalog = pd.DataFrame(reconstructed_catalog, columns=self.cat_cols).astype(self.dtypes)
        # Source-Catalog matching arguments
        ## max distance in degrees
        matching_args = {"max_distance": 7.9, "ReproductionRatio_min": 0.1, "ReproductionRatio_max": 2.5}
        cat_SR_matches = self.match_with_scuba_catalog(self.scuba_cat, reconstructed_catalog, **matching_args, return_df=True)

        # Create Scatter plots for each matched combination to illustrate super-resolution performance on test-test
        self.plot_source_recovery(cat_SR_matches, r"Super-Resolved $500\mu m$ Source Flux $S_{SR}$ [mJy]", os.path.join(self.tdir_out, "scuba_catalog_comparison_recovery_SR_matches.png"))

        # Plot the astrometric accuracy + photometric accuracy
        search_r = self.PS_scuba_plot(cat_SR_matches, rnd_its, os.path.join(self.tdir_out, "PS_Plot.png"), reconstructed_catalog, **matching_args)

        for ID in range(generated_images_model.shape[0]):
            Plot_InputImages(X[ID], os.path.join(self.tdir_out, f"SCUBA2_Comparison_InputHerschelImages_{ID}.png"))
            
            # First pick two highlighted sources/regions from the scuba2 image
            brightest, median_brightest = plot_super_resolved_image(Y[ID], "SCUBA-2 Image", self.target_cat, idx_arr[ID], os.path.join(self.tdir_out, f"SCUBA2Images_{ID}.png"))

            # We have not found two nice regions
            if brightest is None:
                continue

            sources = [brightest, median_brightest]
            
            # Second project these regions on the reconstructed image
            gen_brightest, gen_median_brightest = plot_super_resolved_image(generated_images_model[ID], "Super-Resolved Image", reconstructed_catalog, idx_arr[ID], os.path.join(self.tdir_out, f"ReconstructedSCUBA2Images_{ID}.png"), sources=sources)


if __name__ == "__main__":
    SRModelTest = SRTesterGPU("SCUBA2_comparison.ini")
    SRModelTest.LoadTestData()
    SRModelTest.Analysis()