# Standard library imports
import os
import time
# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import scipy.optimize as opt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from astropy.stats import sigma_clipped_stats
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from photutils.detection import DAOStarFinder
# Custom module imports
from metric_funcs import can_connect_circles, Coincidence_Matches

def confusion_plot(flux_bins, confusion_df_list, label_list, save):
    sigma_instr = 2.
    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins]) * 1000

    # Reliability Plot
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 4))

    # Plot the Reliability and Completeness lines with error bars
    for i in range(len(confusion_df_list)):
        NsamplesR = confusion_df_list[i]['TPr'] + confusion_df_list[i]['FPr']
        NsamplesC = confusion_df_list[i]['TPc'] + confusion_df_list[i]['FNc']
        axs[0].errorbar(flux, confusion_df_list[i]['R'], yerr=np.sqrt(confusion_df_list[i]['R'] * (1- confusion_df_list[i]['R']) / NsamplesR), marker='o', markersize=4,
                        capsize=0, capthick=0, elinewidth=1, label=label_list[i], linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6)

        axs[1].errorbar(flux, confusion_df_list[i]['C'], yerr=np.sqrt(confusion_df_list[i]['C'] * (1- confusion_df_list[i]['C']) / NsamplesC), marker='o', markersize=4,
                        capsize=0, capthick=0, elinewidth=1, label=label_list[i], linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6)

    # Shaded area displaying the uncertainty region of RMS instrumental noise
    axs[0].fill_between(np.array([np.min(flux)-0.1, 4*sigma_instr]), 1.1, color='#B7FFA3', alpha=0.3)
    axs[0].fill_between(np.array([4*sigma_instr, 4.5*sigma_instr]), 1.1, color='#3B5221', alpha=0.3)
    axs[1].fill_between(np.array([np.min(flux)-0.1, 4*sigma_instr]), 1.1, color='#B7FFA3', alpha=0.3)
    axs[1].fill_between(np.array([4*sigma_instr, 4.5*sigma_instr]), 1.1, color='#3B5221', alpha=0.3)

    # Set the x-axis to log scale and add a minor grid
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[0].xaxis.set_tick_params(labelbottom=True)
    axs[0].yaxis.set_tick_params(labelleft=True)
    axs[0].grid(which='both', alpha=0.3, color='lightgrey', linestyle='--')
    axs[1].xaxis.set_tick_params(labelbottom=True)
    axs[1].yaxis.set_tick_params(labelleft=True)
    axs[1].grid(which='both', alpha=0.3, color='lightgrey', linestyle='--')

    # Set the axis labels and title for the main plot
    axs[0].set_xlabel('Generated Source Flux (mJy)', fontsize=8)
    axs[1].set_xlabel('Input Source Flux (mJy)', fontsize=8)

    axs[0].set_ylabel('Reliability', fontsize=8)
    axs[1].set_ylabel('Completeness', fontsize=8)

    # Add a legend with a fancy box to the main plot
    handles, labels = axs[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=True, fancybox=True, shadow=True, fontsize=8, ncol=2)
    legend.get_frame().set_facecolor('white')

    axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[1].set_yticks(np.arange(0, 1.1, 0.1))
    axs[0].tick_params(direction='in', axis='x', which='both', labelsize=6)
    axs[1].tick_params(direction='in', axis='x', which='both', labelsize=6)
    axs[0].tick_params(axis='y', which='both', labelsize=6)
    axs[1].tick_params(axis='y', which='both', labelsize=6)

    axs[0].set_ylim([0., 1.02])
    axs[1].set_ylim([0., 1.02])
    axs[0].set_xlim([np.min(flux)-0.1, np.max(flux)+0.05])
    axs[1].set_xlim([np.min(flux)-0.1, np.max(flux)+0.05])
    axs[0].set_xticks(np.array([1, 10, 100]))
    axs[1].set_xticks(np.array([1, 10, 100]))

    # Create a second axis on top of the axis
    ax1_top = axs[0].twiny()
    ax2_top = axs[1].twiny()
    ax1_top.tick_params(direction='in', axis='x', which='both', labelsize=6)
    ax2_top.tick_params(direction='in', axis='x', which='both', labelsize=6)

    def tick_function(X):
        V = X/sigma_instr
        return ["%.0f" % z for z in V]

    SNR_x1ticks = np.array(axs[0].get_xticks())
    SNR_x2ticks = np.array(axs[1].get_xticks())

    ax1_top.set_xscale('log')
    ax2_top.set_xscale('log')
    ax1_top.set_xlabel('SNR', fontsize=8)
    ax2_top.set_xlabel('SNR', fontsize=8)
    ax1_top.set_xticks(SNR_x1ticks)
    ax1_top.set_xlim(np.array(axs[0].get_xlim())/sigma_instr)
    ax2_top.set_xticks(SNR_x2ticks)
    ax2_top.set_xlim(np.array(axs[1].get_xlim())/sigma_instr)

    # Format the tick labels
    formatter = ticker.FuncFormatter(lambda val, tick_pos: '{:.0f}'.format(val) + r'$\sigma_{inst}$')
    ax1_top.xaxis.set_major_formatter(formatter)
    ax2_top.xaxis.set_major_formatter(formatter)

    # Adjust the spacing between the subplots
    fig.subplots_adjust(hspace=0.3)

    # Save the plot
    fig.savefig(save, dpi=350, edgecolor='white', facecolor='white')


def PS_plot(blind_matches_catalog, rnd_its, save, *catalogs, **matching_args):
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
    binsx = 20
    binsy = 40
    xbins = np.linspace(matching_args['ReproductionRatio_min'], matching_args['ReproductionRatio_max'], binsx + 1)
    # xbins = np.linspace(-25/1000, 25/1000, binsx + 1)

    ybins = np.linspace(0, matching_args['max_distance'], binsy + 1)
    H_blind, xedges_blind, yedges_blind = np.histogram2d(blind_matches_catalog[cols[0]]/blind_matches_catalog[cols[1]], blind_matches_catalog[cols[2]], bins=(xbins, ybins))
    # H_blind, xedges, yedges = np.histogram2d(blind_matches_catalog[cols[0]] - blind_matches_catalog[cols[1]], blind_matches_catalog[cols[2]], bins=(xbins, ybins))

    #H = np.ma.masked_where(H == 0, H)
    x_centers = (xbins[:-1] + xbins[1:]) / 2
    y_centers = (ybins[:-1] + ybins[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute the fake 2d distribution
    H_rnd = np.zeros((binsx, binsy)) # H has x=rows and y=columns, H.T is the transpose
    for i in tqdm(range(rnd_its), desc="Computing the fake 2d distribution..."):
        rnd_matches_catalog = Coincidence_Matches(*catalogs, return_df=True, **matching_args)
        H_it, xedges, yedges = np.histogram2d(rnd_matches_catalog[cols[0]]/rnd_matches_catalog[cols[1]], rnd_matches_catalog[cols[2]], bins=(xbins, ybins))
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
    fig.savefig(save, dpi=400)
    plt.close(fig)

    return np.round(y_centers[index_search_r] * 2)/2



def FluxMatch_Distribution_plot(blind_matches_catalog, rnd_its, save, *catalogs, **matching_args):
    cols = blind_matches_catalog.columns.tolist() if isinstance(blind_matches_catalog, pd.DataFrame) else blind_matches_catalog.keys()
    cmap= plt.get_cmap("turbo")
    cmap.set_bad(color='white')
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.4)
    for i in range(3):
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_aspect('equal', adjustable='box')
        axs[i].tick_params(axis='x', which='both', direction="in", labelsize=8)
        axs[i].tick_params(axis='y', which='both', direction="in", labelsize=8)
        axs[i].yaxis.set_tick_params(labelleft=True)


    # Create the 2D distribution plot
    bins = 50
    xbins = np.logspace(np.log10(1), np.log10(150), bins + 1)/1000
    # xbins = np.linspace(-25/1000, 25/1000, binsx + 1)

    ybins = np.logspace(np.log10(1), np.log10(150), bins + 1)/1000
    H_blind, xedges_blind, yedges_blind = np.histogram2d(blind_matches_catalog[cols[0]], blind_matches_catalog[cols[1]], bins=(xbins, ybins))
    # H_blind, xedges, yedges = np.histogram2d(blind_matches_catalog[cols[0]] - blind_matches_catalog[cols[1]], blind_matches_catalog[cols[2]], bins=(xbins, ybins))

    mask = np.ma.masked_where(H_blind.T == 0, H_blind.T)
    pcol = axs[0].pcolormesh(xedges_blind*1000, yedges_blind*1000, mask, cmap=cmap, vmin=0)#, shading='gouraud')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(pcol, ax=axs[0], cax=cax, location='right')
    cb.set_label('Number of Matches')
    axs[0].set_title("Blind Distribution", fontsize=10)


    x_centers = (xbins[:-1] + xbins[1:]) / 2 * 1000
    y_centers = (ybins[:-1] + ybins[1:]) / 2 * 1000
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute the fake 2d distribution
    H_rnd = np.zeros((bins, bins)) # H has x=rows and y=columns, H.T is the transpose
    for i in tqdm(range(rnd_its), desc="Computing the fake 2d distribution..."):
        rnd_matches_catalog = Coincidence_Matches(*catalogs, return_df=True, **matching_args)
        H_it, xedges, yedges = np.histogram2d(rnd_matches_catalog[cols[0]], rnd_matches_catalog[cols[1]], bins=(xbins, ybins))
        # H_it, xedges, yedges = np.histogram2d(rnd_matches_catalog[cols[0]] - rnd_matches_catalog[cols[1]], rnd_matches_catalog[cols[2]], bins=(xbins, ybins))

        H_rnd += H_it/rnd_its

    mask = np.ma.masked_where(H_rnd.T == 0, H_rnd.T)
    pcol = axs[1].pcolormesh(xedges_blind*1000, yedges_blind*1000, mask, cmap=cmap, vmin=0)#, shading='gouraud')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(pcol, ax=axs[1], cax=cax, location='right')
    cb.set_label('Number of Matches')
    axs[1].set_title("Coincidence Distribution", fontsize=10)

    # Compute the 2D distribution plot of good matches
    H = np.round(H_blind - H_rnd, 0).astype(np.int32)
    H[H<= 0] = 0
    mask = np.ma.masked_where(H.T == 0, H.T)
    pcol = axs[2].pcolormesh(xedges_blind*1000, yedges_blind*1000, mask, cmap=cmap, vmin=0)#, shading='gouraud')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(pcol, ax=axs[2], cax=cax, location='right')
    cb.set_label('Number of Matches')

    axs[2].set_title("Good Distribution", fontsize=10)
    # Set the axis limits for the 2D distribution plot
    axs[0].set_xlim([1, 150])
    axs[0].set_ylim([1, 150])
    # This automatically adjusts the axis limits of all the 2D distribution plots
    # Profile function to fit the 2D distribution plot
    func = lambda x, a, b: a*x + b
    
    for i in range(3):
        xlim = axs[i].get_xlim()
        ylim = axs[i].get_ylim()
        xtrue = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100, base=10)
        # Plot horizontal line equal to min detection threshold generated sources

        axs[i].hlines(y=2.*4, xmin=xlim[0], xmax=xlim[1], linestyle="dashed", color='red', linewidth=1)
        axs[i].vlines(x=2.*4.5, ymin=ylim[0], ymax=ylim[1], linestyle="dashed", color='red', linewidth=1, label="Detection Threshold")

        axs[i].set_ylabel(r"Reconstructed Source Flux  (mJy/beam)", fontsize=10)
        axs[i].set_xlabel(r"Target Source Flux (mJy/beam)", fontsize=10)

        if i == 0 or i == 2:
            axs[i].plot(xtrue, xtrue, color='black', alpha=0.7, label='True Relation', ls='-')
            Target_flux_bins = []
            Reconstructed_flux_bins = []
            for row_idx, row in enumerate(H_blind.T): # y
                for col_idx, counts in enumerate(row): # x
                    if counts > 0:
                        for j in range(int(counts)):
                            Reconstructed_flux_bins.append(Y[row_idx, col_idx])
                            Target_flux_bins.append(X[row_idx, col_idx])

            popt, pcov = curve_fit(func, np.array(Target_flux_bins), np.array(Reconstructed_flux_bins), bounds=([0, -30], [2, 30]))
            axs[i].plot(xtrue, func(xtrue, popt[0], popt[1]), color='black', alpha=0.7, label=f'y={popt[0]:.2f}x + {popt[1]:.2f}', ls='dotted')
        
        legend = axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=True, framealpha=1, fancybox=True, shadow=True, fontsize=8)
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_linewidth(1)


    fig.savefig(save, dpi=400)
    plt.close(fig)
    return H, x_centers, y_centers



def Plot_InputImages(X, save):
    labels = [r"Herschel SPIRE 250 $\mu m$", r"Herschel SPIRE 350 $\mu m$", r"Herschel SPIRE 500 $\mu m$"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.05)
    for i in range(3):
        im = axs[i].imshow(X[i]*1000, origin='lower', cmap='viridis', vmin=0, vmax=75)
        axs[i].set_title(labels[i], fontsize=10)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("top", size="5%", pad=.22)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", label="mJy/beam")
        cax.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.tick_top()
        axs[i].yaxis.set_tick_params(labelleft=True)
    fig.savefig(save, dpi=400)
    plt.close(fig)


def plot_super_resolved_image(Y, title, cat, ImageID, save, sources=None):
    # Find coordinates of brightest and "half" brightest source
    cat_img = cat[cat['ImageID'] == ImageID].reset_index(drop=True)
    WINDOW_SIZE = 40
    Y = np.squeeze(Y)
    compute_y0 = lambda y: y - WINDOW_SIZE//2
    compute_x0 = lambda x: x - WINDOW_SIZE//2
    i = 0

    # If sources is not None then we have been given a source/region to highlight
    # Find the coordinates of these sources
    if sources is not None:
        r_brightest = np.sqrt((sources[0]['xpix'] - cat_img['xpix'])**2 + (sources[0]['ypix'] - cat_img['ypix'])**2)
        r_median_brightest = np.sqrt((sources[1]['xpix'] - cat_img['xpix'])**2 + (sources[1]['ypix'] - cat_img['ypix'])**2) 
        rmin_brightest_idx = np.argmin(r_brightest)
        rmin_median_brightest_idx = np.argmin(r_median_brightest)

        # We do not impose a maximum distance, so one can just choose from stored images later that are good
        gen_brightest = cat_img.iloc[rmin_brightest_idx]
        gen_median_brightest = cat_img.iloc[rmin_median_brightest_idx]

    # The highlighted regions should be away from the border of the image being bad comparisons
    if sources is None:
        while True:
            brightest = cat_img.sort_values(by='peak', ascending=False).iloc[0 + i]
            median_brightest = cat_img.sort_values(by='peak', ascending=False).iloc[cat_img.shape[0]//4 - i]

            if compute_y0(brightest["ypix"]) > 0 and compute_x0(brightest["xpix"]) > 0 and (compute_y0(brightest["ypix"]) + WINDOW_SIZE) < Y.shape[1] and (compute_x0(brightest["xpix"]) + WINDOW_SIZE) < Y.shape[0]:
                if compute_y0(median_brightest["ypix"]) > 0 and compute_x0(median_brightest["xpix"]) > 0 and (compute_y0(median_brightest["ypix"]) + WINDOW_SIZE) < Y.shape[1] and (compute_x0(median_brightest["xpix"]) + WINDOW_SIZE) < Y.shape[0]:
                    break
            if i == 10:
                return (None, None)
            i += 1
    else:
        brightest = sources[0]
        median_brightest = sources[1]

    fig = plt.figure(figsize=(8, 6))
    grid = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[3, 1], wspace=0.05, hspace=0.2)

    axs = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1])
    ]    
    colors = ['#FFD43E', '#45E0A5']
    # highlighted areas
    

    y0 = [compute_y0(brightest["ypix"]), compute_y0(median_brightest["ypix"])]
    ysize = [40, 40]
    x0 = [compute_x0(brightest["xpix"]), compute_x0(median_brightest["xpix"])]
    xsize = [40, 40]

    # Main plot
    im1 = axs[0].imshow(Y*1000, origin='lower', cmap='viridis', vmin=0, vmax=30, aspect='equal')
    axs[0].set_title(title, fontsize=10)
    # cax = divider.append_axes("top", size="5%", pad=.22)
    cbar = fig.colorbar(im1, ax=axs[0], orientation="vertical", label="mJy/beam", location='left')
    # cax.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.xaxis.tick_top()

    # Draw highlighted areas
    for i in range(2):
        rect = plt.Rectangle((x0[i], y0[i]), xsize[i], ysize[i], fill=False, edgecolor=colors[i])
        axs[0].add_patch(rect)

    axs[0].tick_params(axis='x', which='both', labelsize=8)
    axs[0].tick_params(axis='y', which='both', labelsize=8)

    # Highlighted plots
    pos = axs[0].get_position()
    for i in range(2):
        axs[i+1].imshow(Y*1000, origin='lower', cmap='viridis', vmin=0, vmax=30, aspect='equal')
        axs[i+1].set_xlim(x0[i], x0[i]+xsize[i])
        axs[i+1].set_ylim(y0[i], y0[i]+ysize[i])
        xlim = axs[i+1].get_xlim()
        ylim = axs[i+1].get_ylim()
        pos2 = axs[i+1].get_position()
        axs[i+1].set_xticks(np.arange(np.round(xlim[0]), np.round(xlim[1])+10, 10))
        axs[i+1].set_yticks(np.arange(np.round(ylim[0]), np.round(ylim[1])+10, 10))
        if i == 0:
            axs[i+1].set_position([pos.x0, pos2.y0, pos2.width, pos2.height])
            axs[i+1].scatter(brightest['xpix'], brightest['ypix'], marker="x", s=10, color='red', linewidths=0.5)
            if sources is not None:
                axs[i+1].scatter(gen_brightest['xpix'], gen_brightest['ypix'], marker="x", s=10, color='blue', linewidths=0.5)

        else:
            axs[i+1].set_position([(pos.x0 + pos.width) - pos2.width, pos2.y0, pos2.width, pos2.height])
            axs[i+1].scatter(median_brightest['xpix'], median_brightest['ypix'], marker="x", s=10, color='red', linewidths=0.5)
            if sources is not None:
                axs[i+1].scatter(gen_median_brightest['xpix'], gen_median_brightest['ypix'], marker="x", s=10, color='blue', linewidths=0.5)

        axs[i+1].spines['left'].set_color(colors[i])
        axs[i+1].spines['bottom'].set_color(colors[i])
        axs[i+1].spines['right'].set_color(colors[i])
        axs[i+1].spines['top'].set_color(colors[i])
        axs[i+1].spines['left'].set_linewidth(2)
        axs[i+1].spines['bottom'].set_linewidth(2)
        axs[i+1].spines['right'].set_linewidth(2)
        axs[i+1].spines['top'].set_linewidth(2)

        axs[i+1].tick_params(axis='x', which='both', labelsize=8)
        axs[i+1].tick_params(axis='y', which='both', labelsize=8)

        # indicate relevant source position

    fig.savefig(save, dpi=400)
    plt.close(fig)

    if sources is not None:
        return gen_brightest, gen_median_brightest
    return brightest, median_brightest


def source_profile_comparison(Y_list, sources_arr, instr_noise, save):
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 4.5), nrows=1, ncols=2, sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.2)

    labels = ["Horizontal True Profile", "Horizontal Generated Profile", "Model2: Horizontal Generated Profile"]
    colors = ["red", "green", "blue"]
    h_slice = np.arange(-10, 10+1, 1)

    for i in range(len(Y_list)):
        Y = np.squeeze(Y_list[i])
        profile_brightest = Y[int(np.round(sources_arr[0][i]['ypix']))-10:int(np.round(sources_arr[0][i]['ypix'])) + 10 + 1, int(np.round(sources_arr[0][i]['xpix']))]

        profile_median_brightest = Y[int(np.round(sources_arr[1][i]['ypix']))-10:int(np.round(sources_arr[1][i]['ypix'])) + 10 + 1, int(np.round(sources_arr[1][i]['xpix']))]

        ax1.plot(h_slice, profile_brightest*1000, color=colors[i], alpha=0.7, ls='-', marker='o', fillstyle='none', label=labels[i])
        ax2.plot(h_slice, profile_median_brightest*1000, color=colors[i], alpha=0.7, ls='-', marker='o', fillstyle='none', label=labels[i])

    ax1.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax2.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
        
    ax1.hlines(y=instr_noise*1000, xmin=-11, xmax=11, color='black', ls='-.', label=r"$\sigma_{inst}$" + f'={instr_noise*1000:.1f} mJy')
    ax2.hlines(y=instr_noise*1000, xmin=-11, xmax=11, color='black', ls='-.', label=r"$\sigma_{inst}$" + f'={instr_noise*1000:.1f} mJy')

    ax1.tick_params(axis='x', which='both', labelsize=8)
    ax1.tick_params(axis='y', which='both', labelsize=8)
    ax2.tick_params(axis='x', which='both', labelsize=8)
    ax2.tick_params(axis='y', which='both', labelsize=8)
    # ax3.scatter(0, strueprof_peak*1000, marker='x', color='blue', label='True Peak Flux', s=15)
    # ax3.scatter(0, sgenprof_peak*1000, marker='x', color='green', label='Generated Peak Flux', s=15)
    ax1.set_xlabel("Offset ['']", fontsize=8)
    ax1.set_ylabel(r'$S_{500}$ [mJy\beam]', fontsize=8)

    ax2.set_xlabel("Offset ['']", fontsize=8)
    ax2.set_ylabel(r'$S_{500}$ [mJy\beam]', fontsize=8)

    ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper right', fontsize=6)
    ax2.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper right', fontsize=6)


    ax1.set_xlim([h_slice[0], h_slice[-1]])
    ax2.set_xlim([h_slice[0], h_slice[-1]])

    ax1.set_ylim([0, ax1.get_ylim()[1] + 15])
    ax2.set_ylim([0, ax2.get_ylim()[1] + 10])

    fig.savefig(save, dpi=400)
    plt.close(fig)


def FluxReproduction_plot(counts, x_centers, y_centers, sigma_instr, save):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    labels = ["Current Model", "Model2"]
    for i in range(len(counts)):
        mean = []
        std_err = []

        for x_idx, Sin in enumerate(x_centers):
            tmp = []
            for y_idx, Sout in enumerate(y_centers):
                if counts[i][x_idx,y_idx] >= 0:
                    tmp += [(Sin-Sout)/Sin] *int(counts[i][x_idx,y_idx])
            if len(tmp) == 0:
                mean.append(-10)
                std_err.append(-10)
            else:
                mean.append(np.mean(tmp))
                std_err.append(np.std(tmp))

        mean = np.array(mean)
        std_err = np.array(std_err)

        ax.errorbar(x_centers[np.where(mean != -10)], mean[np.where(mean != -10)], yerr=std_err[np.where(mean != -10)], marker='o', markersize=4, 
                capsize=0, capthick=0, elinewidth=1, label=labels[i], linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6)
    
    # Set the x-axis to log scale and add a minor grid
    ax.set_xscale('log')
    ax.grid(which='both', alpha=0.4, color='lightgrey', linestyle='--')

    # Set the axis labels and title for the main plot
    ax.set_xlabel('Target Source Flux (mJy)', fontsize=8)
    ax.set_ylabel(r'$\frac{S_{target, 500} - S_{reconstructed, 500}}{S_{target, 500}}$', fontsize=8)

    # Add a legend with a fancy box to the main plot
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=True, fancybox=True, shadow=True, fontsize=8, ncol=2)
    legend.get_frame().set_facecolor('white')

    ax.tick_params(direction='in', axis='x', which='both', labelsize=6)

    ax.tick_params(axis='y', which='both', labelsize=6)

    # Create a second axis on top of the axis
    ax1_top = ax.twiny()
    ax1_top.tick_params(direction='in', axis='x', which='both', labelsize=6)
    #axs[1, 0].tick_params(direction='in', axis='x', which='both', labelsize=6)

    def tick_function(X):
        V = X/(sigma_instr*1000)
        return ["%.0f" % z for z in V]
   
    SNR_x1ticks = np.array(ax.get_xticks())

    ax1_top.set_xscale('log')
    ax1_top.set_xlabel('SNR', fontsize=8)
    ax1_top.set_xticks(SNR_x1ticks)
    ax1_top.set_xlim(np.array(ax.get_xlim())/(sigma_instr*1000))

    # Format the tick labels
    formatter = ticker.FuncFormatter(lambda val, tick_pos: '{:.0f}'.format(val) + r'$\sigma_{inst}$')
    ax1_top.xaxis.set_major_formatter(formatter)


    ## Save the plot
    fig.savefig(save, dpi=350, edgecolor='white', facecolor='white')
    plt.show()
    plt.close(fig)

