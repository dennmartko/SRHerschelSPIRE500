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
from matplotlib.cm import ScalarMappable
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

        ## Paths with COSMOS PRIOR data
        self.path_input = self.config['COMMON']['path_herschel_images'].rstrip().lstrip()
        # self.correlation_data_path = correlation_data_path 
        ## Indicate purpose of run
        self.RUN_NAME = self.config['MODELS']['model_name'].rstrip().lstrip()

        ## Load path to models dir
        self.models_lib_path = self.config['COMMON']['model_outdir'].rstrip().lstrip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)
        self.kind = self.config['MODELS']['model_type'].rstrip().lstrip()

        ## Paths to jin et al catalog
        self.path_jin_cat = self.config['COMMON']['path_jinetal_catalog'].rstrip().lstrip()

        ## Set classes
        self.classes = [i.strip(' ') for i in self.config['COMMON']['input'].rstrip().lstrip().split(",")]

        self.TOTAL_SAMPLES = len([entry for entry in os.listdir(os.path.join(self.path_input , self.classes[0]))])
        # self.TOTAL_SAMPLES_CORR = len([entry for entry in os.listdir(os.path.join(self.correlation_data_path, self.classes[0]))])
        self.tdir_out = os.path.join(self.model_path, f"results_{self.kind}")
        if not os.path.isdir(self.tdir_out):
            os.mkdir(self.tdir_out)
    def LoadTestData(self):
        self.TEST_BATCH_SIZE = 6

        self.test_arr_X = np.zeros((self.TOTAL_SAMPLES, 4, 106, 106))

        self.wcs_arr = []
        self.wcs_arr_250 = []
        for i in tqdm(range(self.TOTAL_SAMPLES), desc=f"Loading Data From {self.path_input }"):
            for k in range(len(self.classes)):
                with fits.open(os.path.join(self.path_input , f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
                    self.test_arr_X[i][k] = hdu[0].data
                    if self.classes[k] == "250":
                        # We have to compute the corresponding header of the super-resolved images
                        ## Copy header
                        self.wcs_arr_250.append(WCS(hdu[0].header))
                        new_header = hdu[0].header.copy()
                        pix_scale = abs(hdu[0].header["PC1_1"]*3600)

                        scaler = pix_scale/1

                        # New header values, note I only use this to obtain the correct CRPIX
                        w_hdu = WCS(hdu[0].header)
                        new_crpix = w_hdu[::1/scaler, ::1/scaler].to_header()

                        # Rescale header for WCS
                        new_header["CRPIX1"] = new_crpix["CRPIX1"]
                        new_header["CRPIX2"] = new_crpix["CRPIX2"]
                        new_header["PC1_1"] = float(hdu[0].header["PC1_1"])/(scaler)
                        new_header["PC2_2"] = float(hdu[0].header["PC2_2"])/(scaler)

                        new_header["CDELT1"] = new_header["PC1_1"]
                        new_header["CDELT2"] = new_header["PC2_2"]

                        new_header["NAXIS1"] = 424
                        new_header["NAXIS2"] = 424

                        del new_header["PC1_1"]
                        del new_header["PC2_2"]

                        self.wcs_arr.append(WCS(new_header))

        self.dtypes = {"peak_mjy": np.float32, "xpix": np.float32, "ypix":np.float32, "ra":np.float32, "dec":np.float32, "ImageID":np.int32}
        self.cat_cols = ["peak_mjy", "xpix", "ypix", "ra", "dec", "ImageID"]
        # Load in the Simulation master catalog
        
        self.jin_cat = pd.DataFrame(fits.open(self.path_jin_cat, memmap=False)[1].data)
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

            # jin_converted = self.wcs_arr[ImageIDList[i]].wcs_world2pix(self.jin_cat[self.jin_cat["F500"] >= 2][['RA', 'DEC']].values, 0)
            # jin_flux = self.jin_cat[self.jin_cat["F500"] >= 2]["F500"].values
            # jin_flux = jin_flux[jin_converted[:,0] > 0]
            # jin_converted = jin_converted[jin_converted[:,0] > 0]
            # jin_flux = jin_flux[jin_converted[:,1] > 0]
            # jin_converted = jin_converted[jin_converted[:,1] > 0]
            # jin_flux = jin_flux[jin_converted[:,0] < 424]
            # jin_converted = jin_converted[jin_converted[:,0] < 424]
            # jin_flux = jin_flux[jin_converted[:,1] < 424]
            # jin_converted = jin_converted[jin_converted[:,1] < 424]

            # print(jin_converted.shape)
            # print(jin_flux.shape)

            # plt.imshow(img_batch[i], vmin=0, vmax=30/1000)
            # plt.scatter(tr[:, 0], tr[:, 1], marker='o', facecolors='none', edgecolors='red', s=30)
            # plt.scatter(jin_converted[:, 0], jin_converted[:, 1], marker='o', facecolors='none', edgecolors='orange', s=30)
            # for idx, source in enumerate(sources):
            #     plt.text(x=tr[idx, 0], y=tr[idx, 1], s=f"{source['peak']*1000:.0f}", color='red')
            # for idx in range(jin_converted.shape[0]):
            #     plt.text(x=jin_converted[idx, 0], y=jin_converted[idx, 1], s=f"{jin_flux[idx]:.0f}", color='orange')
            # plt.show()

            # jin_converted = self.wcs_arr[ImageIDList[i]].wcs_world2pix(self.jin_cat[self.jin_cat["F500"] >= 2][['RA', 'DEC']].values, 0)
            # jin_converted = jin_converted[jin_converted[:,0] > 0]
            # jin_converted = jin_converted[jin_converted[:,1] > 0]
            # jin_converted = jin_converted[jin_converted[:,0] < 424]
            # jin_converted = jin_converted[jin_converted[:,1] < 424]

            for idx, source in enumerate(sources):
                cat["peak_mjy"].append(source["peak"]*1000)
                cat["xpix"].append(source["xcentroid"])
                cat["ypix"].append(source["ycentroid"])
                cat["ra"].append(tr_world[idx][0])
                cat["dec"].append(tr_world[idx][1])
                cat["ImageID"].append(ImageIDList[i])
        return cat


    def match_with_jin_catalog(self, reconstructed_catalog, max_distance=4, ReproductionRatio_min=0.5, ReproductionRatio_max=1.5, return_df=False):
        matches_catalog = {"Jin et al Catalog Source Flux": [], "Jin et al Catalog Source Flux err": [], "Detected Source Flux": [], "Distance" : []}
        ## Iterate over target catalog
        ## Find the best match
        ## Fill the matches catalog
        self.matched_sr_jin_catalog = self.jin_cat.copy()
        self.matched_sr_jin_catalog["SR_F500"] = -99.0
        self.matched_sr_jin_catalog["SR_F500_dist"] = -99.0
        for detected_source_idx, detected_source in reconstructed_catalog.iterrows():
            r = np.sqrt((detected_source['ra'] - self.jin_cat['RA'].values)**2 + (detected_source['dec'] - self.jin_cat['DEC'].values)**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx]*3600 <= max_distance: #and ReproductionRatio_min <= Reconstructed_catalog[mask_generated]['peak'].values[rmin_idx]/target_source['peak'] <= ReproductionRatio_max:
                self.matched_sr_jin_catalog.at[rmin_idx, "SR_F500"] = detected_source['peak_mjy']
                self.matched_sr_jin_catalog.at[rmin_idx, "SR_F500_dist"] = r[rmin_idx]*3600
                # matches_catalog["Detected Source Flux"].append(detected_source['peak_mjy'])
                # matches_catalog["Jin et al Catalog Source Flux"].append(self.jin_cat['F500'].values[rmin_idx])
                # matches_catalog["Jin et al Catalog Source Flux err"].append(self.jin_cat['DF500'].values[rmin_idx])
                # matches_catalog["Distance"].append(r[rmin_idx]*3600) # In arcseconds
    
    def plot_number_counts(self, catalog_sr, min_flux_mjy=2, save_path=None):
        # Area both catalogs
        area_sr = (np.max(catalog_sr['ra']) - np.min(catalog_sr['ra'])) * (np.max(catalog_sr['dec']) - np.min(catalog_sr['dec']))

        catalog_sr = catalog_sr[catalog_sr['peak_mjy'] >= min_flux_mjy]
        catalog_jin = self.jin_cat[(self.jin_cat['RA'] >= np.min(catalog_sr['ra'])) & (self.jin_cat['RA'] <= np.max(catalog_sr['ra']))]
        catalog_jin = catalog_jin[(catalog_jin['DEC'] >= np.min(catalog_sr['dec'])) & (catalog_jin['DEC'] <= np.max(catalog_sr['dec']))]
        catalog_jin = catalog_jin[catalog_jin['F500'] >= min_flux_mjy]

        num_bins = np.linspace(min_flux_mjy, np.max(catalog_sr['peak_mjy']), 100)
        
        # Create a logarithmic y-scale histogram plot
        plt.figure(figsize=(8, 6))
        plt.hist(catalog_sr['peak_mjy'], bins=num_bins, alpha=0.7, edgecolor='blue', histtype='step', label='This work', lw=1.5)

        plt.hist(catalog_jin['F500'], bins=num_bins, alpha=0.7, edgecolor='red', histtype='step', label='Jin et al.', lw=1.5)

        plt.yscale('log')  # Set y-axis scale to logarithmic
        plt.xlabel(r'Herschel SPIRE $500\mu m$ Source Flux [mJy]', fontsize=12)
        plt.ylabel('Counts (N(S))', fontsize=12)
        plt.tick_params(axis='both', direction='in', which='both')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()

        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        # Show the plot
        plt.close()

    def plot_flux_comparison(self, catalog_sr, min_flux_mjy=2, save_path=None):
        only_matches_cat = self.matched_sr_jin_catalog[(self.matched_sr_jin_catalog["SR_F500"] >= min_flux_mjy) & (self.matched_sr_jin_catalog["F500"] >= min_flux_mjy)]
        flux_sr_500 = only_matches_cat["SR_F500"]
        flux_jin_500 = only_matches_cat["F500"]
        bin_width = 5  # Adjust the bin width as needed
        bin_edges = np.logspace(np.log10(min_flux_mjy), np.log10(np.max(flux_jin_500)), 10, base=10)
        bin_indices_x = np.digitize(flux_jin_500, bin_edges)
        bin_indices_y = np.digitize(flux_sr_500, bin_edges)

        bin_medians_x = [np.median(flux_jin_500[bin_indices_x == i]) for i in range(1, len(bin_edges))]
        bin_medians_y = [np.median(flux_sr_500[bin_indices_x == i]) for i in range(1, len(bin_edges))]
        bin_std_errors_x = [np.std(flux_jin_500[bin_indices_x == i]) for i in range(1, len(bin_edges))]
        bin_std_errors_y = [np.std(flux_sr_500[bin_indices_x == i]) for i in range(1, len(bin_edges))]

        catalog_sr = catalog_sr[catalog_sr['peak_mjy'] >= min_flux_mjy]
        catalog_jin = self.jin_cat[(self.jin_cat['RA'] >= np.min(catalog_sr['ra'])) & (self.jin_cat['RA'] <= np.max(catalog_sr['ra']))]
        catalog_jin = catalog_jin[(catalog_jin['DEC'] >= np.min(catalog_sr['dec'])) & (catalog_jin['DEC'] <= np.max(catalog_sr['dec']))]
        catalog_jin = catalog_jin[catalog_jin['F500'] >= min_flux_mjy]

        N500sources_jin = catalog_jin.shape[0]
        N500sources_thiswork = catalog_sr.shape[0]

        print(catalog_sr.shape)

        plt.figure(figsize=(8, 6))
        plt.scatter(flux_jin_500, flux_sr_500, color='blue', alpha=0.9, label='Matches', s=2)

        plt.errorbar(bin_medians_x, bin_medians_y, xerr=bin_std_errors_x, yerr=bin_std_errors_y, fmt='.', color='red', label='Median with SE', markersize=5, capsize=3)

        plt.ylabel(r'$500\mu m$ flux [mJy] (this work)', fontsize=12)
        plt.xlabel(r'$500\mu m$ flux [mJy] (Jin et al. Catalog)', fontsize=12)
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        x_true = np.linspace(min_flux_mjy, np.max(flux_jin_500), 1000)
        plt.plot(x_true, x_true, linestyle='dashed', color='red')
        plt.title(r"$N_{jin\  et \ al}$" + f"= {N500sources_jin}" + r", $N_{this \ work}$" +  f"= {N500sources_thiswork}" + r", $N_{matches}$" +  f"= {len(flux_jin_500)}", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        plt.close()

    def write_catalog_complementation(self, catalog_sr, min_flux_mjy, save):
        catalog_sr = catalog_sr[catalog_sr['peak_mjy'] >= min_flux_mjy]
        catalog_jin = self.jin_cat[(self.jin_cat['RA'] >= np.min(catalog_sr['ra'])) & (self.jin_cat['RA'] <= np.max(catalog_sr['ra']))]
        catalog_jin = catalog_jin[(catalog_jin['DEC'] >= np.min(catalog_sr['dec'])) & (catalog_jin['DEC'] <= np.max(catalog_sr['dec']))]
        catalog_jin = catalog_jin[catalog_jin['F500'] >= min_flux_mjy]

        N500sources_jin = catalog_jin.shape[0]
        N500sources_thiswork = catalog_sr.shape[0]
        with open(save, "w") as f:
            f.write(f"Number of 500 micron (>2mJy) Sources; Jin et al: {N500sources_jin}\n")
            f.write(f"Number of 500 micron (>2mJy) Sources; This work: { N500sources_thiswork}\n")

        print(f"Number of 500 micron (>2mJy) Sources; Jin et al: {N500sources_jin}\n")
        print(f"Number of 500 micron (>2mJy) Sources; This work: { N500sources_thiswork}\n")

    def Analysis(self):
        self.LoadModel(kind=self.kind, model_path=self.model_path)
        # Generated Source Catalog needs to be filled first  
        reconstructed_catalog = {"peak_mjy": [], "xpix": [], "ypix": [], "ra":[], "dec":[], "ImageID": []}
        its = self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE if self.test_arr_X.shape[0] > self.TEST_BATCH_SIZE else 1
        print(self.jin_cat.columns)

        rnd_its = 10
        for batch_idx in tqdm(range(its), desc="Super-Resolving Test Data with main model!"):
            # Load the batch of data
            X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            
            # Compute the corresponding ImageIDs
            idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
            ImageIDList = idx_arr.tolist()

            # Super-Resolve the batch of test data
            gen_valid = self.generator(X, training=False).numpy()

            # Fill the generated catalog with detected sources
            reconstructed_catalog = self.fill_worldcatalog(gen_valid, reconstructed_catalog, ImageIDList, self.instr_noise)

            # plt.imshow(np.squeeze(gen_valid[4]), vmin=0, vmax=30/1000)
            # plt.show()
        # Convert the generated catalog to a Pandas DataFrame
        reconstructed_catalog = pd.DataFrame(reconstructed_catalog, columns=self.cat_cols).astype(self.dtypes)
        jin_converted = self.wcs_arr[ImageIDList[3]].wcs_world2pix(self.jin_cat[self.jin_cat["F500"] >= 2][['RA', 'DEC']].values, 0)
        jin_converted = jin_converted[jin_converted[:,0] > 0]
        jin_converted = jin_converted[jin_converted[:,1] > 0]
        jin_converted = jin_converted[jin_converted[:,0] < 424]
        jin_converted = jin_converted[jin_converted[:,1] < 424]

        sources = self.wcs_arr[ImageIDList[3]].wcs_world2pix(reconstructed_catalog[reconstructed_catalog["peak_mjy"] >= 2][['ra', 'dec']].values, 0)
        sources = sources[sources[:,0] > 0]
        sources = sources[sources[:,1] > 0]
        sources = sources[sources[:,0] < 424]
        sources = sources[sources[:,1] < 424]

        plt.imshow(np.squeeze(gen_valid[3])*1000, vmin=0, vmax=10, cmap='viridis')
        sm = ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=10))
        sm.set_array([])  # This step is important to set the data range for the colorbar
        plt.colorbar(sm, label="mJy/beam")
        plt.scatter(sources[:, 0], sources[:, 1], marker='o', facecolors='none', edgecolors='red', s=50, label="This work")
        plt.scatter(jin_converted[:, 0], jin_converted[:, 1], marker='o', facecolors='none', edgecolors='orange', s=50, label="Jin et al.")
        plt.title("Super-resolved Image", fontsize=10)
        plt.legend(fontsize=10, loc='lower left')
        plt.savefig(os.path.join(self.tdir_out, "COSMOS_catalog_projection.png"), dpi=350, bbox_inches='tight')
        plt.show()


        self.match_with_jin_catalog(reconstructed_catalog, max_distance=4, ReproductionRatio_min=0.5, ReproductionRatio_max=1.5, return_df=True)

        # Plot number counts for Herschel for this work and Jin et al.
        self.plot_number_counts(reconstructed_catalog, min_flux_mjy=2, save_path=os.path.join(self.tdir_out, "COSMOS_counts_comparison.pdf"))

        # Flux comparison with Jin et al.
        self.plot_flux_comparison(reconstructed_catalog, min_flux_mjy=2, save_path=os.path.join(self.tdir_out, "COSMOS_F500_comparison.pdf"))

        # Txt diagnostics for this work and Jin et al. Particularly hits/missed number counts
        self.write_catalog_complementation(reconstructed_catalog, min_flux_mjy=2, save=os.path.join(self.tdir_out, "COSMOS_catalog_improvement.txt"))

        # Finally, write the catalog files (The SR catalog, and the SR catalog matched with Jin et al. catalog)
        table_sr = fits.BinTableHDU.from_columns([fits.Column(name=col, format='D', array=reconstructed_catalog[col]) for col in reconstructed_catalog.columns])
        table_sr_jin = fits.BinTableHDU.from_columns([fits.Column(name=col, format='D', array=self.matched_sr_jin_catalog[col]) for col in self.matched_sr_jin_catalog.columns])

        primary_header = fits.Header()
        hdu_list_sr = fits.HDUList([fits.PrimaryHDU(header=primary_header), table_sr])
        hdu_list_sr_jin = fits.HDUList([fits.PrimaryHDU(header=primary_header), table_sr_jin])
        hdu_list_sr.writeto(os.path.join(self.tdir_out, "COSMOS_SR_catalog.fits"), overwrite=True)
        hdu_list_sr_jin.writeto(os.path.join(self.tdir_out, "COSMOS_SR+Jin_catalog.fits"), overwrite=True)
        
if __name__ == "__main__":
    SRModelTest = SRTesterGPU("SRHerschelSPIRE.ini")
    SRModelTest.LoadTestData()
    SRModelTest.Analysis()