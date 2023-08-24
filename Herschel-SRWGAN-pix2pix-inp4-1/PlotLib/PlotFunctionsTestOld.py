# Standard library imports
import os

# Third-party library imports
import numpy as np
import seaborn as sns
import tensorflow as tf
import scipy.optimize as opt

from astropy.stats import sigma_clipped_stats
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from photutils.detection import DAOStarFinder

# Custom module imports
from metric_funcs import can_connect_circles

def plot_kernelprofile(Y, gen_valid, kernel_size, save):
    def TwoDGaussian(XY, x0, y0, sigma_x, sigma_y, amp, offset):
        x, y = XY
        gaus = offset + amp*np.exp(-((x-x0)**2 / (2*sigma_x**2)  + (y-y0)**2/(2*sigma_y**2)))
        return gaus.ravel()

    assert (kernel_size % 2) == 1, "Kernel size should be odd!"
    # Initialise a 21x21 kernel to compute source profiles
    # Note the kernel should be odd for symmetric gaussian profiles
    kernel_target = np.zeros((kernel_size, kernel_size))
    kernel_gen = np.zeros((kernel_size, kernel_size))

    # Whenever possible we use the x and y coordinates of detected sources
    # We center the kernel on these coordinates and add the pixel coordinates
    # Finally the kernel is normalized and a 2D gaussian profile is fitted
    Y = np.squeeze(Y)
    gen_valid = np.squeeze(gen_valid)
    _, __, std_noise = sigma_clipped_stats(Y, sigma=3, maxiters=50)
    #std_noise = 2.8/1000 # Jy
    source_finder = DAOStarFinder(fwhm=7.9, threshold=4.5*std_noise)

    # Fill catalog for Target with true sources
    target_sources = source_finder(Y)
    gen_sources = source_finder(Y)
    try:
        tr = np.transpose((target_sources['xcentroid'], target_sources['ycentroid']))
        tr = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
    except:
        return # No sources detected

    for source in target_sources:
        try:
            kernel_target += Y[int(np.round(source['ycentroid'])) - int((kernel_size - 1)/2):int(np.round(source['ycentroid'])) + int((kernel_size - 1)/2) + 1, int(np.round(source['xcentroid'])) - int((kernel_size - 1)/2):int(np.round(source['xcentroid'])) + int((kernel_size - 1)/2) + 1]
        except:
            continue
    for source in gen_sources:
        try:
            kernel_gen += gen_valid[int(np.round(source['ycentroid'])) - int((kernel_size - 1)/2):int(np.round(source['ycentroid'])) + int((kernel_size - 1)/2) + 1, int(np.round(source['xcentroid'])) - int((kernel_size - 1)/2):int(np.round(source['xcentroid'])) + int((kernel_size - 1)/2) + 1]
        except:
            continue

    kernel_target /= np.max(kernel_target)
    kernel_gen /= np.max(kernel_gen)

    x = np.arange(0, kernel_size, 1)
    y = np.arange(0, kernel_size, 1)
    x, y = np.meshgrid(x, y)

    # initial guess
    if np.isnan(kernel_target[0,0]) or np.isnan(kernel_gen[0,0]): return # Security
    p0 = ((kernel_size - 1)/2, (kernel_size - 1)/2, 3.5, 3.5, 1., 0.)
    popt_target, pcov_target = opt.curve_fit(TwoDGaussian, (x, y), kernel_target.ravel(), p0=p0,
                                             bounds=((0, 0, 1, 1, 0.5, -0.1), (kernel_size/2, kernel_size/2, 10., 10., 5., 1.)))
    popt_gen, pcov_gen = opt.curve_fit(TwoDGaussian, (x, y), kernel_gen.ravel(), p0=p0,
                                             bounds=((0, 0, 1, 1, 0.5, -0.1), (kernel_size/2, kernel_size/2, 10., 10., 5., 1.)))
    
    FWHM_x_target = np.abs(2*popt_target[2]*np.sqrt(2*np.log(2)))
    FWHM_y_target = np.abs(2*popt_target[3]*np.sqrt(2*np.log(2)))

    FWHM_x_gen = np.abs(2*popt_gen[2]*np.sqrt(2*np.log(2)))
    FWHM_y_gen = np.abs(2*popt_gen[3]*np.sqrt(2*np.log(2)))

    # Figure
    fig, axs = plt.subplots(2, 3, figsize=(24,10))

    im1 = axs[0, 0].imshow(kernel_target, vmin=0, vmax=1, cmap="viridis")
    im2 = axs[1, 0].imshow(kernel_gen, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im1, ax=axs[0,0], location='top', label="Normalized Flux ratio", pad=0.1)
    fig.colorbar(im2, ax=axs[1, 0], location='top', label="Normalized Flux ratio", pad=0.1)
    
    G2D = lambda x, y, x0, y0, sigma_x, sigma_y, amp, offset: offset + amp*np.exp(-((x-x0)**2 / (2*sigma_x**2)  + (y-y0)**2/(2*sigma_y**2)))
    X = np.linspace(0, kernel_size, 100)
    Y = np.linspace(0, kernel_size, 100)

    axs[0, 1].plot(np.arange(0, kernel_size, 1), kernel_target[int((kernel_size - 1)/2), :], label="Target", color='black', alpha=0.9)
    axs[0, 1].plot(X, G2D(X, int(np.round(popt_target[1])), popt_target[0], popt_target[1], popt_target[2], popt_target[3], popt_target[4], popt_target[5]),  label=rf"Gaussian Fit $\sigma_x = {popt_target[2]:.2f} \pm {np.sqrt(pcov_target[2][2]):.2f}$", color='red', alpha=0.9, linestyle='dashed')
    axs[0, 1].set_title(f"Target FWHM (x): {FWHM_x_target:.2f} pixels", fontsize=12)

    axs[0, 2].plot(np.arange(0, kernel_size, 1), kernel_target[:, int((kernel_size - 1)/2)], label="Target", color='black', alpha=0.9)
    axs[0, 2].plot(Y, G2D(int(np.round(popt_target[0])) , Y, popt_target[0], popt_target[1], popt_target[2], popt_target[3], popt_target[4], popt_target[5]),  label=rf"Gaussian Fit $\sigma_y = {popt_target[3]:.2f} \pm {np.sqrt(pcov_target[3][3]):.2f}$", color='red', alpha=0.9, linestyle='dashed')
    axs[0, 2].set_title(f"Target FWHM (y): {FWHM_y_target:.2f} pixels", fontsize=12)

    axs[1, 1].plot(np.arange(0, kernel_size, 1), kernel_gen[int((kernel_size - 1)/2), :], label="Generated", color='black', alpha=0.9)
    axs[1, 1].plot(X, G2D(X, int(np.round(popt_gen[1])) , popt_gen[0], popt_gen[1], popt_gen[2], popt_gen[3], popt_gen[4], popt_gen[5]),  label=rf"Gaussian Fit $\sigma_x = {popt_gen[2]:.2f} \pm {np.sqrt(pcov_gen[2][2]):.2f}$", color='red', alpha=0.9, linestyle='dashed')
    axs[1, 1].set_title(f"FWHM (x): {FWHM_x_gen:.2f} pixels", fontsize=12)

    axs[1, 2].plot(np.arange(0, kernel_size, 1), kernel_gen[:, int((kernel_size - 1)/2)], label="Generated", color='black', alpha=0.9)
    axs[1, 2].plot(Y, G2D(int(np.round(popt_gen[0])), Y, popt_gen[0], popt_gen[1], popt_gen[2], popt_gen[3], popt_gen[4], popt_gen[5]),  label=rf"Gaussian Fit $\sigma_y = {popt_gen[3]:.2f} \pm {np.sqrt(pcov_gen[3][3]):.2f}$", color='red', alpha=0.9, linestyle='dashed')
    axs[1, 2].set_title(f"FWHM (y): {FWHM_y_gen:.2f} pixels", fontsize=12)

    for i in range(2):
        for j in range(3):
            axs[i, j].legend()
            if j > 0:
                axs[i, j].set_ylim([0, 1])
                axs[i, j].set_xlim([0, 20])
                axs[i, j].set_ylabel("Normalized Flux")
                if j == 1:
                    axs[i, j].set_xlabel("x (pixel)")
                if j == 2:
                    axs[i, j].set_xlabel("y (pixel)")

    fig.savefig(save, dpi=350)
    plt.close('all')

def insetplot(X, gen, target, Y_cat, save):
    gen = np.squeeze(gen)
    target = np.squeeze(target)

    ## estimate noise level
    _, __, std_noise = sigma_clipped_stats(target, sigma=3, maxiters=50)
    #print(np.std(target.flatten()[target.flatten() <= 0]))

    fig = plt.figure(figsize=(24,6))
    gs = GridSpec(1, 5 , hspace=0.005)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3:])

    im1 = ax1.imshow(X[0]*1000, origin='lower', cmap='viridis', vmin=0, vmax=75)
    im2 = ax2.imshow(target*1000, origin='lower', cmap='viridis',  vmin=0, vmax=30)
    im3 = ax3.imshow(gen*1000, origin='lower', cmap='viridis',  vmin=0, vmax=30)

    fig.colorbar(im1, ax=ax1, location='top', label="mJy/beam", pad=0.1)
    fig.colorbar(im2, ax=ax2, location='top', label="mJy/beam", pad=0.1)
    fig.colorbar(im3, ax=ax3, location='top', label="mJy/beam", pad=0.1)
    ax1.set_title(r"Herschel SPIRE 250 $\mu m$", fontsize=13, weight='bold', alpha=.8)
    ax2.set_title(r"Herschel SPIRE 500 $\mu m$ Super-Resolved", fontsize=13, weight='bold', alpha=.8)
    ax3.set_title(r"Reconstructed SR Herschel SPIRE 500 $\mu m$", fontsize=13, weight='bold', alpha=.8)

    # Make a zoom
    y0 = 100
    ysize = 40
    x0 = 60
    xsize = 40

    y0_inp = y0/4
    ysize_inp = ysize/4
    x0_inp = x0/4
    xsize_inp = xsize/4

    axins1 = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins3 = ax3.inset_axes([0.5, 0.5, 0.47, 0.47])


    #sources = Y_cat[np.logical_and(y0 < Y_cat[:,1] < y0+ysize, x0 < Y_cat[:,0] < x0+xsize)]
    sources = Y_cat[np.where((Y_cat[:,2] < y0+ysize) & (Y_cat[:,1] < x0+xsize))]
    sources = sources[np.where((sources[:,2] > y0) & (sources[:,1] > x0))][:,0:-1]
    try:
        sprof_idx = np.argmax(sources[:,0])
    except:
        plt.close('all')
        return

    if len(sources) == 0:
        return

    source_finder = DAOStarFinder(fwhm=7.9, threshold=4.*std_noise)
    gen_sources = source_finder(gen)
    try:
        tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
    except:
        plt.close('all')
        return
    tr_gen = tr_gen[np.where((tr_gen[:,-1] < y0+ysize) & (tr_gen[:,-2] < x0+xsize))]
    tr_gen = tr_gen[np.where((tr_gen[:,-1] > y0) & (tr_gen[:,-2] > x0))]
    if len(tr_gen) == 0:
        plt.close('all')
        return
    # sources[:,0] -= xsize
    # sources[:,1] -= ysize

    axins1.imshow(X[0], origin='lower', cmap='viridis', vmin=0, vmax=75/1000)
    axins2.imshow(target, origin='lower', cmap='viridis', vmin=0, vmax=75/1000)
    axins3.imshow(gen, origin='lower', cmap='viridis', vmin=0, vmax=75/1000)

    Flag = False
    for idx, s in enumerate(sources):
        if idx == sprof_idx:
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')
            axins3.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')

            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] > 8:
                plt.close('all')
                return
        else:
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')
            axins3.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')

    for idx, s in enumerate(tr_gen):
        peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        if idx == rmin_idx:
            r_flag = np.sqrt((s[0] - sources[:,1])**2 + (s[1] - sources[:,2])**2)
            # Reliability Issue Detection
            potential_flags = np.where(r_flag <= 8)[0]
            if len(potential_flags) > 1:
                true_sources_in_fhwm = sources[:,1:][potential_flags]
                # Try to connect each true source under the PSF 'umbrella' of a generated source
                # If this is possible, we have no issue and the generated source is "fine"
                # If this is not possible, there is a source-source seperation > PSF. 
                # These should not be under the PSF of the generated source, and we should have atleast 2 seperate generated sources.
                if not can_connect_circles(true_sources_in_fhwm, 8):
                    # Flag found!
                    Flag = True
            axins3.scatter(s[0], s[1], marker="s", s=peak*10000, color='green', linewidths=1, facecolors='none')
            sgenprof_peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        else:
            axins3.scatter(s[0], s[1], marker="s", s=peak*10000, color='red', linewidths=1, facecolors='none')


    axins1.set_xlim(x0_inp, x0_inp+xsize_inp)
    axins1.set_ylim(y0_inp, y0_inp+ysize_inp)
    axins2.set_xlim(x0, x0+xsize)
    axins2.set_ylim(y0, y0+ysize)
    axins3.set_xlim(x0, x0+xsize)
    axins3.set_ylim(y0, y0+ysize)

    axins1.set_xticklabels('')
    axins1.set_yticklabels('')
    axins2.set_xticklabels('')
    axins2.set_yticklabels('')
    axins3.set_xticklabels('')
    axins3.set_yticklabels('')

    if Flag:
        axins1.spines['bottom'].set_color('green')
        axins1.spines['top'].set_color('green') 
        axins1.spines['right'].set_color('green')
        axins1.spines['left'].set_color('green')

        axins2.spines['bottom'].set_color('green')
        axins2.spines['top'].set_color('green') 
        axins2.spines['right'].set_color('green')
        axins2.spines['left'].set_color('green')

        axins3.spines['bottom'].set_color('green')
        axins3.spines['top'].set_color('green')
        axins3.spines['right'].set_color('green')
        axins3.spines['left'].set_color('green')
    else:
        axins1.spines['bottom'].set_color('gray')
        axins1.spines['top'].set_color('gray') 
        axins1.spines['right'].set_color('gray')
        axins1.spines['left'].set_color('gray')

        axins2.spines['bottom'].set_color('gray')
        axins2.spines['top'].set_color('gray') 
        axins2.spines['right'].set_color('gray')
        axins2.spines['left'].set_color('gray')

        axins3.spines['bottom'].set_color('gray')
        axins3.spines['top'].set_color('gray')
        axins3.spines['right'].set_color('gray')
        axins3.spines['left'].set_color('gray')

    ax1.indicate_inset_zoom(axins1, edgecolor='gray')
    ax2.indicate_inset_zoom(axins2, edgecolor='gray')
    ax3.indicate_inset_zoom(axins3, edgecolor='gray')


    # Profile flux of a source
    strueprof_x = sources[sprof_idx][1]
    strueprof_y = sources[sprof_idx][2]
    strueprof_peak = sources[sprof_idx][0]

    if std_noise > strueprof_peak or std_noise > sgenprof_peak:
        plt.close('all')
        return

    sgenprof_x = tr_gen[rmin_idx][0]
    sgenprof_y = tr_gen[rmin_idx][1]
    
    h_slicetrue = np.arange(int(np.round(strueprof_x, decimals=0)) - 10, int(np.round(strueprof_x, decimals=0)) + 10, 1)
    h_slicegen = np.arange(int(np.round(sgenprof_x, decimals=0)) - 10, int(np.round(sgenprof_x, decimals=0)) + 10, 1)

    ax4.plot((h_slicegen - int(sgenprof_x))*1, gen[int(sgenprof_y), h_slicegen]*1000, color='green', alpha=0.7, ls='dashed', marker='o', fillstyle='none', label="Generated Profile")
    ax4.plot((h_slicetrue - int(strueprof_x))*1, target[int(strueprof_y), h_slicetrue]*1000, color='blue', alpha=0.9, ls='-', marker='o', fillstyle='none', label="True Profile")
    ax4.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax4.xaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax4.hlines(y=std_noise*1000, xmin=(h_slicetrue[0]- int(strueprof_x))*1, xmax=(h_slicetrue[-1]- int(strueprof_x))*1, color='black', ls='dotted', label=rf'Noise $\sigma={std_noise*1000:.1f}$ mJy')
    # ax3.scatter(0, strueprof_peak*1000, marker='x', color='blue', label='True Peak Flux', s=15)
    # ax3.scatter(0, sgenprof_peak*1000, marker='x', color='green', label='Generated Peak Flux', s=15)
    ax4.set_xlabel("Offset ('')")
    ax4.set_ylabel(r'$S$ (mJy\beam)')

    ax4.set_xlim([(h_slicetrue[0]- int(strueprof_x))*1, (h_slicetrue[-1]- int(strueprof_x))*1])

    ax4.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper right')
    # if Flag:
    #     ax1.text(x=0.05, y=0.84, s="""FLAGGED""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure, color="green")

    # ax1.text(x=0.05, y=0.88, s=rf"""Generated Source detection threshold is set at $3\sigma={1000*std_noise*3:.1f}mJy/beam$""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    # ax1.text(x=0.05, y=0.86, s=rf"""Target Source detection threshold is set at $5\sigma={1000*std_noise*5:.1f}mJy/beam$""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    # ax1.text(x=0.05, y=0.9, s="Comparison Plot", ha='left', fontsize=13, weight='bold', alpha=.8, transform=fig.transFigure)
    plt.savefig(save, dpi=350)
    if Flag:
        print("flag!")
    plt.close('all')

def insetplot_diff(X, gen, target, Y_cat, save):
    gen = np.squeeze(gen)
    target = np.squeeze(target)

    ## estimate noise level
    _, __, std_noise = sigma_clipped_stats(target, sigma=3, maxiters=50)
    #print(np.std(target.flatten()[target.flatten() <= 0]))

    fig = plt.figure(figsize=(24,6))
    gs = GridSpec(1, 5 , hspace=0.005)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3:])

    im1 = ax1.imshow(target*1000, origin='lower', cmap='viridis', vmin=0, vmax=75)
    im2 = ax2.imshow(target*1000 - gen*1000, origin='lower', cmap='viridis',  vmin=0, vmax=30)
    im3 = ax3.imshow(tf.abs(target*1000 - gen*1000), origin='lower', cmap='viridis',  vmin=0, vmax=30)

    fig.colorbar(im1, ax=ax1, location='top', label="mJy/beam", pad=0.1)
    fig.colorbar(im2, ax=ax2, location='top', label="mJy/beam", pad=0.1)
    fig.colorbar(im3, ax=ax3, location='top', label="mJy/beam", pad=0.1)
    ax1.set_title(r"Herschel SPIRE 500 Super-Resolved $\mu m$", fontsize=13, weight='bold', alpha=.8)
    ax2.set_title(r"Difference", fontsize=13, weight='bold', alpha=.8)
    ax3.set_title(r"Absolute Difference", fontsize=13, weight='bold', alpha=.8)

    # Make a zoom
    y0 = 100
    ysize = 40
    x0 = 60
    xsize = 40

    y0_inp = y0/4
    ysize_inp = ysize/4
    x0_inp = x0/4
    xsize_inp = xsize/4

    axins1 = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins3 = ax3.inset_axes([0.5, 0.5, 0.47, 0.47])


    #sources = Y_cat[np.logical_and(y0 < Y_cat[:,1] < y0+ysize, x0 < Y_cat[:,0] < x0+xsize)]
    sources = Y_cat[np.where((Y_cat[:,2] < y0+ysize) & (Y_cat[:,1] < x0+xsize))]
    sources = sources[np.where((sources[:,2] > y0) & (sources[:,1] > x0))][:,0:-1]
    try:
        sprof_idx = np.argmax(sources[:,0])
    except:
        plt.close('all')
        return

    if len(sources) == 0:
        return

    source_finder = DAOStarFinder(fwhm=7.9, threshold=3*std_noise)
    gen_sources = source_finder(gen)
    try:
        tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
    except:
        plt.close('all')
        return
    tr_gen = tr_gen[np.where((tr_gen[:,-1] < y0+ysize) & (tr_gen[:,-2] < x0+xsize))]
    tr_gen = tr_gen[np.where((tr_gen[:,-1] > y0) & (tr_gen[:,-2] > x0))]
    if len(tr_gen) == 0:
        plt.close('all')
        return
    # sources[:,0] -= xsize
    # sources[:,1] -= ysize

    axins1.imshow(target, origin='lower', cmap='viridis', vmin=0, vmax=75/1000)
    axins2.imshow(target - gen, origin='lower', cmap='viridis', vmin=0, vmax=75/1000)
    axins3.imshow(tf.abs(gen - target), origin='lower', cmap='viridis', vmin=0, vmax=75/1000)

    Flag = False
    for idx, s in enumerate(sources):
        if idx == sprof_idx:
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')
            axins3.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')

            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] > 8:
                plt.close('all')
                return
        else:
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')
            axins3.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')

    for idx, s in enumerate(tr_gen):
        peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        if idx == rmin_idx:
            r_flag = np.sqrt((s[0] - sources[:,1])**2 + (s[1] - sources[:,2])**2)
            # Reliability Issue Detection
            potential_flags = np.where(r_flag <= 8)[0]
            if len(potential_flags) > 1:
                true_sources_in_fhwm = sources[:,1:][potential_flags]
                # Try to connect each true source under the PSF 'umbrella' of a generated source
                # If this is possible, we have no issue and the generated source is "fine"
                # If this is not possible, there is a source-source seperation > PSF. 
                # These should not be under the PSF of the generated source, and we should have atleast 2 seperate generated sources.
                if not can_connect_circles(true_sources_in_fhwm, 8):
                    # Flag found!
                    Flag = True
            axins3.scatter(s[0], s[1], marker="s", s=peak*10000, color='green', linewidths=1, facecolors='none')
            sgenprof_peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        else:
            axins3.scatter(s[0], s[1], marker="s", s=peak*10000, color='red', linewidths=1, facecolors='none')


    axins1.set_xlim(x0_inp, x0_inp+xsize_inp)
    axins1.set_ylim(y0_inp, y0_inp+ysize_inp)
    axins2.set_xlim(x0, x0+xsize)
    axins2.set_ylim(y0, y0+ysize)
    axins3.set_xlim(x0, x0+xsize)
    axins3.set_ylim(y0, y0+ysize)

    axins1.set_xticklabels('')
    axins1.set_yticklabels('')
    axins2.set_xticklabels('')
    axins2.set_yticklabels('')
    axins3.set_xticklabels('')
    axins3.set_yticklabels('')

    if Flag:
        axins1.spines['bottom'].set_color('green')
        axins1.spines['top'].set_color('green') 
        axins1.spines['right'].set_color('green')
        axins1.spines['left'].set_color('green')

        axins2.spines['bottom'].set_color('green')
        axins2.spines['top'].set_color('green') 
        axins2.spines['right'].set_color('green')
        axins2.spines['left'].set_color('green')

        axins3.spines['bottom'].set_color('green')
        axins3.spines['top'].set_color('green')
        axins3.spines['right'].set_color('green')
        axins3.spines['left'].set_color('green')
    else:
        axins1.spines['bottom'].set_color('gray')
        axins1.spines['top'].set_color('gray') 
        axins1.spines['right'].set_color('gray')
        axins1.spines['left'].set_color('gray')

        axins2.spines['bottom'].set_color('gray')
        axins2.spines['top'].set_color('gray') 
        axins2.spines['right'].set_color('gray')
        axins2.spines['left'].set_color('gray')

        axins3.spines['bottom'].set_color('gray')
        axins3.spines['top'].set_color('gray')
        axins3.spines['right'].set_color('gray')
        axins3.spines['left'].set_color('gray')

    ax1.indicate_inset_zoom(axins1, edgecolor='gray')
    ax2.indicate_inset_zoom(axins2, edgecolor='gray')
    ax3.indicate_inset_zoom(axins3, edgecolor='gray')


    # Profile flux of a source
    strueprof_x = sources[sprof_idx][1]
    strueprof_y = sources[sprof_idx][2]
    strueprof_peak = sources[sprof_idx][0]

    if std_noise > strueprof_peak or std_noise > sgenprof_peak:
        plt.close('all')
        return

    sgenprof_x = tr_gen[rmin_idx][0]
    sgenprof_y = tr_gen[rmin_idx][1]
    
    h_slicetrue = np.arange(int(np.round(strueprof_x, decimals=0)) - 10, int(np.round(strueprof_x, decimals=0)) + 10, 1)
    h_slicegen = np.arange(int(np.round(sgenprof_x, decimals=0)) - 10, int(np.round(sgenprof_x, decimals=0)) + 10, 1)

    ax4.plot((h_slicegen - int(sgenprof_x))*1, gen[int(sgenprof_y), h_slicegen]*1000, color='green', alpha=0.7, ls='dashed', marker='o', fillstyle='none', label="Generated Profile")
    ax4.plot((h_slicetrue - int(strueprof_x))*1, target[int(strueprof_y), h_slicetrue]*1000, color='blue', alpha=0.9, ls='-', marker='o', fillstyle='none', label="True Profile")
    ax4.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax4.xaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax4.hlines(y=std_noise*1000, xmin=(h_slicetrue[0]- int(strueprof_x))*1, xmax=(h_slicetrue[-1]- int(strueprof_x))*1, color='black', ls='dotted', label=rf'Noise $\sigma={std_noise*1000:.1f}$ mJy')
    # ax3.scatter(0, strueprof_peak*1000, marker='x', color='blue', label='True Peak Flux', s=15)
    # ax3.scatter(0, sgenprof_peak*1000, marker='x', color='green', label='Generated Peak Flux', s=15)
    ax4.set_xlabel("Offset ('')")
    ax4.set_ylabel(r'$S$ (mJy\beam)')

    ax4.set_xlim([(h_slicetrue[0]- int(strueprof_x))*1, (h_slicetrue[-1]- int(strueprof_x))*1])

    ax4.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper right')
    # if Flag:
    #     ax1.text(x=0.05, y=0.84, s="""FLAGGED""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure, color="green")

    # ax1.text(x=0.05, y=0.88, s=rf"""Generated Source detection threshold is set at $3\sigma={1000*std_noise*3:.1f}mJy/beam$""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    # ax1.text(x=0.05, y=0.86, s=rf"""Target Source detection threshold is set at $5\sigma={1000*std_noise*5:.1f}mJy/beam$""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    # ax1.text(x=0.05, y=0.9, s="Comparison Plot", ha='left', fontsize=13, weight='bold', alpha=.8, transform=fig.transFigure)
    plt.savefig(save, dpi=350)
    if Flag:
        print("flag!")
    plt.close('all')

def FluxReproduction_plot(bins, mean_relSdiff, mean_err_relSdiff, MAD_relSdiff, MAD_err_relSdiff, flux_type, save):
    # 1 row, 2 col plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].plot(bins, mean_relSdiff)
    axs[0].fill_between(bins, mean_relSdiff - mean_err_relSdiff, mean_relSdiff + mean_err_relSdiff, alpha=0.3)

    axs[1].plot(bins, MAD_relSdiff)
    axs[1].fill_between(bins, MAD_relSdiff - MAD_err_relSdiff, MAD_relSdiff + MAD_err_relSdiff, alpha=0.3)


    for i in range(2):
        if flux_type == 'peak':
            axs[i].set_xlabel(r"Source Flux (mJy/beam)")
        elif flux_type == 'aper':
            axs[i].set_xlabel(r"Source Flux (mJy/pixel)")

    if flux_type == 'peak':
        axs[0].set_ylabel(r"$\left<\left|S_{peak, in} - S_{peak, out}\right|/S_{peak, in}\right>$")
    elif flux_type == 'aper':
        axs[0].set_ylabel(r"$\left<\left|S_{aper in} - S_{aper, out}\right|/S_{aper, in}\right>$")
    axs[1].set_ylabel(r"Average Relative Spread")

    xlim = axs[0].get_xlim()
    for i in range(2):
        axs[i].hlines(y=0.0, xmin=xlim[0], xmax=xlim[1], linestyles='dashed', color='black', alpha=0.9)
  
    fig.savefig(save, dpi=350, edgecolor='white', facecolor='white')


def hexplot(matches_df_arr, flux_type, save):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    #fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharey=True)
    for i in range(3):
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')

    nbins = 50

    x_bins = np.logspace(np.log10(1), np.log10(150), nbins + 1)
    y_bins = np.logspace(np.log10(1), np.log10(150), nbins + 1)

    c1, xedges1, yedges1, img1 = axs[0].hist2d(np.array(matches_df_arr[0]['Sin_true'])*1000, np.array(matches_df_arr[0]['Sout1'])*1000, bins=(x_bins, y_bins), cmap='inferno')
    cbar1 = fig.colorbar(img1, ax=axs[0], label="Number of Matches")
    cbar1.ax.tick_params(labelsize=6)

    # Calculate the average of the distribution
    c2 = np.zeros((nbins, nbins))
    for i in range(len(matches_df_arr)):
        H, _, __ = np.histogram2d(np.array(matches_df_arr[i]['Sin_rnd'])*1000, np.array(matches_df_arr[i]['Sout2'])*1000, bins=(x_bins, y_bins))
        c2 += H/len(matches_df_arr)

    pc2 = axs[1].pcolormesh(xedges1, yedges1, np.round(c2.T, 0), cmap="inferno", vmin=0)
    cbar2 = fig.colorbar(pc2, ax=axs[1], label="Number of Matches")
    cbar2.ax.tick_params(labelsize=6)

    c3 = np.round(c1 - c2, 0)

    #print(c3.shape, np.array(matches_df['Sin_true']).shape)
    #hexplot3 = axs[2].hexbin(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, C=hist3, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
    # c3, xedges, yedges, img3 = axs[2].hist2d(bins=(x_edges, y_edges), weights=c3.flatten(), cmap='inferno')
    pc3 = axs[2].pcolormesh(xedges1, yedges1, c3.T, cmap="inferno", vmin=0)
    cbar3 = fig.colorbar(pc3, ax=axs[2], label="Number of Matches")
    cbar3.ax.tick_params(labelsize=6)

    for i in range(3):
        xlim = axs[i].get_xlim()
        ylim = axs[i].get_ylim()
        xtrue = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), nbins, base=10)
        # Plot horizontal line equal to min detection threshold generated sources

        #ytrue = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 100, base=10)
        if flux_type == 'peak':
            axs[i].hlines(y=2.8, xmin=xlim[0], xmax=xlim[1], linestyle="dashed", color='#3CB043', linewidth=1)
            axs[i].vlines(x=2.8*5, ymin=ylim[0], ymax=ylim[1], linestyle="dashed", color='#3CB043', linewidth=1)

            axs[i].set_ylabel(r"Peak Source Flux Super-Resolved $500\mu m$ (mJy/beam)")
            axs[i].set_xlabel(r"Peak Source Flux Herschel SPIRE $500\mu m$ (mJy/beam)")
        elif flux_type == 'aper':
            aper2flux_factor = 2.8
            axs[i].set_ylabel(r"Aperture Flux Super-Resolved $500\mu m$ (mJy/pixel)")
            axs[i].set_xlabel(r"Aperture Flux Herschel SPIRE $500\mu m$ (mJy/pixel)")

            axs[i].hlines(y=2.8 * 1/aper2flux_factor, xmin=xlim[0], xmax=xlim[1], linestyle="dashed", color='#3CB043', linewidth=1)
            axs[i].vlines(x=2.8*5 * 1/aper2flux_factor, ymin=ylim[0], ymax=ylim[1], linestyle="dashed", color='#3CB043', linewidth=1)



        axs[i].plot(xtrue, xtrue, color='red', alpha=0.7, label='True correlation', ls='dashed')

    fig.savefig(save, dpi=350)
    plt.close('all')


def PS_plot(matches_df_arr, flux_type, save):
    titles = ["All Matches", "Random Matches", "Coincidence Matches"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    Sdiff_all = np.array(matches_df_arr[0]['Sin_true'])*1000 - np.array(matches_df_arr[0]['Sout1'])*1000
    nbins = 50
    x_bins = np.linspace(-20, 20, nbins + 1)
    y_bins = np.linspace(0, 8, nbins + 1)

    c1, xedges1, yedges1, img1 = axs[0].hist2d(Sdiff_all, np.array(matches_df_arr[0]['true_offset']), bins=(x_bins, y_bins), cmap='inferno')
    cbar1 = fig.colorbar(img1, ax=axs[0], label="Number of Matches")
    cbar1.ax.tick_params(labelsize=6)

    # Calculate the average of the distribution
    c2 = np.zeros((nbins, nbins))
    for i in range(len(matches_df_arr)):
        Sdiff_rnd = np.array(matches_df_arr[i]['Sin_rnd'])*1000 - np.array(matches_df_arr[i]['Sout2'])*1000
        # print(Sdiff_rnd.shape, np.array(matches_df_arr[i]['rnd_offset']).shape)
        H, _, __ = np.histogram2d(Sdiff_rnd, np.array(matches_df_arr[i]['rnd_offset']), bins=(x_bins, y_bins))
        c2 += H/len(matches_df_arr)
        

    pc2 = axs[1].pcolormesh(xedges1, yedges1, np.round(c2.T, 0), cmap="inferno", vmin=0)
    cbar2 = fig.colorbar(pc2, ax=axs[1], label="Number of Matches")
    cbar2.ax.tick_params(labelsize=6)

    c3 = np.round(c1 - c2, 0)
    pc3 = axs[2].pcolormesh(xedges1, yedges1, c3.T, cmap="inferno", vmin=0)
    cbar3 = fig.colorbar(pc3, ax=axs[2], label="Number of Matches")
    cbar3.ax.tick_params(labelsize=6)

    for i in range(3):
        axs[i].set_ylabel(r"Offset ('')")
        if flux_type == 'peak':
            axs[i].set_xlabel(r"$S_{peak, in} - S_{peak, out}$ (mJy/beam)")
        elif flux_type == 'aper':
            axs[i].set_xlabel(r"$S_{aper, in} - S_{aper, out}$ (mJy/pixel)")
        axs[i].set_title(titles[i])

    fig.savefig(save, dpi=350)
    plt.close('all')


def confusion_plot(flux_bins, confusion_df, save):
    plt.clf()
    C_color = "#1e1e1e"
    
    # Completeness
    fig = plt.figure(figsize=(9.5, 4.5), frameon=True, facecolor='white')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey=ax1)

    axs = [ax1, ax2]
    
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins])
    ax1.plot(flux*1000, confusion_df['C'], alpha=0.9, lw=1)
    ax2.plot(flux*1000, confusion_df['R'], alpha=0.9, lw=1, label="Reliability")
    ax2.plot(flux*1000,  confusion_df['flag_R'], alpha=0.9, lw=1, label="RFLAG")

    for idx, ax in enumerate(axs):
        if idx == 0:
            ax.set_xlabel(r"Herschel SPIRE $500\mu m$ Source Flux (mJy/beam)")
        elif idx == 1:
            ax.set_xlabel(r"Generated Herschel SPIRE $500\mu m$ Source Flux (mJy/beam)")
        
        # Plot Styling
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_xscale('log')

        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(axis='y', length=0)
        for tick_label in ax.yaxis.get_ticklabels():
            tick_label.set_fontsize(11)
        for tick_label in ax.xaxis.get_ticklabels():
            tick_label.set_fontsize(11)
            tick_label.set_horizontalalignment('center')
        ax.tick_params(direction='out', axis='x', length=7, color='0.1')

    ax1.set_title(u'Completeness: Accuracy', color=C_color, fontstyle='italic')
    ax2.set_title(u'Reliability: Precision', color=C_color, fontstyle='italic')
    ax2.legend(loc="best")
    ## Save the plot
    fig.savefig(save, dpi=350, edgecolor='white', facecolor='white')


def confusion_plot_old(flux_bins, confusion_df, save):
    plt.clf()
    C_color = "#1e1e1e"
    R_color = "#3399FF"
    flag_R_color = "#F47174"

    fig = plt.figure(figsize=(9, 5), frameon=True, facecolor='white')
    ax1 = fig.add_subplot(111)
    
    # Gridline Styling
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    # Completeness and Reliability line plots
    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins])
    ax1.plot(flux*1000, confusion_df['C'], color=C_color, alpha=0.9, label='Completeness', marker='o', lw=2, fillstyle='none')
    ax1.plot(flux*1000, confusion_df['R'], color=R_color, alpha=0.9, label='Reliability', marker='o', lw=2, fillstyle='none')
    ax1.plot(flux*1000, confusion_df['flag_R'], color=flag_R_color, alpha=0.9, label='Flag_R', marker='o', lw=2, fillstyle='none')

    ax1.set_xlabel("Source Flux (mJy/beam)")

    # Plot Styling
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.set_xscale('log')
    #ax1.set_xticks(np.arange(0, flux[-10]*1000, 5))
    #ax1.set_xticks(np.arange(0, 125, 5))
    #ax1.set_xlim([0, flux[-1]*1000])
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')

    ax1.tick_params(axis='y', length=0)
    for tick_label in ax1.yaxis.get_ticklabels():
        tick_label.set_fontsize(11)
    for tick_label in ax1.xaxis.get_ticklabels():
        tick_label.set_fontsize(11)
        tick_label.set_horizontalalignment('center')
    ax1.tick_params(direction='in', axis='x', length=7, color='0.1')

    # Display analogy completeness and reliability
    ax1.annotate(u'Completeness: Accuracy', xy=(0.09, 0.9),
                xycoords='figure fraction', size=12, color=C_color,
                fontstyle='italic')
    ax1.annotate(u'Reliability: Precision', xy=(0.75, 0.9),
                xycoords='figure fraction', size=12, color=R_color,
                fontstyle='italic')
    ax1.annotate(u'Reliability FLAG', xy=(0.09, 0.01),
                xycoords='figure fraction', size=12, color=flag_R_color,
                fontstyle='italic')
    ax1.text(x=-1.5, y=1.15, s="Confusion Plot", ha='left', fontsize=13, weight='bold', alpha=.8)
    # Save the plot
    fig.savefig(save, dpi=350, edgecolor='white', facecolor='white')

# def completeness_plot(aper_flux_bins, c, α, save):
#     color = "#30a2da"
#     plt.figure(figsize=(7,7))
#     plt.plot(np.array([((bin[0] + bin[1])/2) for bin in aper_flux_bins]), c, color=color, alpha=0.9, label=rf"$\alpha = ${α}", marker="s", lw=2)
#     plt.title("Completeness (SNR>=5) - Matched generated sources", fontsize=15)
#     plt.xlabel("Peak Flux (Jy/beam)", fontsize=15)
#     plt.ylabel("Completeness", fontsize=15)
#     plt.legend()
#     plt.savefig(save)
#     plt.close()

# def reliability_plot(aper_flux_bins, r, α, save):
#     color = "#30a2da"
#     plt.figure(figsize=(7,7))
#     plt.plot(np.array([((bin[0] + bin[1])/2) for bin in aper_flux_bins]), r, color=color, alpha=0.9, label=rf"$\alpha = ${α}", marker="s", lw=2)
#     plt.title("Reliability (SNR >= 1) - Matched real sources", fontsize=15)
#     plt.xlabel("Peak Flux (Jy/beam)", fontsize=15)
#     plt.ylabel("Reliability", fontsize=15)
#     plt.legend()
#     plt.savefig(save)
#     plt.close()

def comparison_plot(gen_output, target, save):
    plt.clf()
    fig = plt.figure(figsize=(14,14))
    fig.suptitle("Comparison generated-validation images (@500microns, 1'', 7.9'' FWHM)")
    gs = GridSpec(4, 2, hspace=.4, wspace=0.1)
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(gs[j, i])
            if i == 0:
                ax.imshow(gen_output[j], aspect="auto", cmap="gnuplot2", vmin = 0)
                ax.set_title("Generated")
            else:
                ax.imshow(target[j], aspect="auto", cmap="gnuplot2", vmin = 0)
                ax.set_title("Validation images")

    plt.savefig(save)
    plt.close()

def hexplot_old(matches_df, save1, save2):
    mean_relSdiff_true = []
    std_relSdiff_true = []

    mean_relSdiff_rnd = []
    std_relSdiff_rnd = []

    mean_relSdiff_flag = []
    std_relSdiff_flag = []
    fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharey=True)
    #fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharey=True)
    for i in range(3):
        for j in range(2):
            axs[j,i].set_xscale('log')
            axs[j,i].set_yscale('log')

    for i in range(2):
        x_bins = np.logspace(np.log10(1), np.log10(150), 50)
        y_bins = np.logspace(np.log10(1), np.log10(150), 50)
        #hist1, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, bins=(50,50))
        c1, xedges1, yedges1, img1 = axs[i,0].hist2d(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, bins=(x_bins, y_bins), cmap='inferno')
        xcenters1 = (xedges1[:-1] + xedges1[1:]) / 2
        ycenters1 = (yedges1[:-1] + yedges1[1:]) / 2
        #hexplot1 = axs[0].hexbin(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, C=hist1, cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
        cbar1 = fig.colorbar(img1, ax=axs[i,0], label="Number of Matches")
        cbar1.ax.tick_params(labelsize=6)


        if i == 0:
            for x_idx, Sin in enumerate(xcenters1):
                tmp = []
                for y_idx, Sout in enumerate(ycenters1):
                    tmp += [abs(Sin-Sout)/Sin for k in range(int(c1[x_idx,y_idx]))]
                if len(tmp) == 0:
                    mean_relSdiff_true.append(-1)
                    std_relSdiff_true.append(0)
                else:
                    mean_relSdiff_true.append(np.mean(tmp))
                    std_relSdiff_true.append(np.std(tmp))


        if i == 0:
            #hist2, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_edges, y_edges))
            c2, xedges2, yedges2, img2 = axs[i,1].hist2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_bins, y_bins), cmap='inferno')
            #hexplot2 = axs[1].hexbin(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, C=hist2, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
            cbar2 = fig.colorbar(img2, ax=axs[i,1], label="Number of Matches")
            cbar2.ax.tick_params(labelsize=6)
        elif i == 1:
            #hist2, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_edges, y_edges))
            c2, xedges2, yedges2, img2 = axs[i,1].hist2d(np.array(matches_df['Sin_flag'])*1000, np.array(matches_df['Sout3'])*1000, bins=(x_bins, y_bins), cmap='inferno')
            #hexplot2 = axs[1].hexbin(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, C=hist2, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
            cbar2 = fig.colorbar(img2, ax=axs[i,1], label="Number of Matches")
            cbar2.ax.tick_params(labelsize=6)

        c3 = c1 - c2
        for x_idx, Sin in enumerate(xcenters1):
            tmp = []
            for y_idx, Sout in enumerate(ycenters1):
                if c3[x_idx,y_idx] >= 0:
                    tmp += [abs(Sin-Sout)/Sin for k in range(int(c3[x_idx,y_idx]))]
            if i == 0:
                if len(tmp) == 0:
                    mean_relSdiff_rnd.append(-1)
                    std_relSdiff_rnd.append(0)
                else:
                    mean_relSdiff_rnd.append(np.mean(tmp))
                    std_relSdiff_rnd.append(np.std(tmp))
            if i == 1:
                if len(tmp) == 0:
                    mean_relSdiff_flag.append(-1)
                    std_relSdiff_flag.append(0)
                else:
                    mean_relSdiff_flag.append(np.mean(tmp))
                    std_relSdiff_flag.append(np.std(tmp))

        #print(c3.shape, np.array(matches_df['Sin_true']).shape)
        #hexplot3 = axs[2].hexbin(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, C=hist3, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
        # c3, xedges, yedges, img3 = axs[2].hist2d(bins=(x_edges, y_edges), weights=c3.flatten(), cmap='inferno')
        pc = axs[i,2].pcolormesh(xedges2, yedges2, c3.T, cmap="inferno", vmin=0)
        cbar3 = fig.colorbar(pc, ax=axs[i,2], label="Number of Matches")
        cbar3.ax.tick_params(labelsize=6)


    for i in range(3):
        for j in range(2):
            xlim = axs[j,i].get_xlim()
            #ylim = axs[j,i].get_ylim()
            xtrue = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 50, base=10)
            # Plot horizontal line equal to min detection threshold generated sources

            axs[j, i].hlines(y=3.8, xmin=xlim[0], xmax=xlim[1], linestyle="dashed", color='#3CB043', linewidth=1)
            #ytrue = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 100, base=10)
            axs[j,i].set_ylabel(r"$S_{out}$ (mJy/beam)")
            axs[j,i].set_xlabel(r"$S_{in}$ (mJy/beam)")
            axs[j,i].plot(xtrue, xtrue, color='red', alpha=0.7, label='True correlation', ls='dashed')

    fig.savefig(save1, dpi=350)
    plt.close('all')

    # relative flux difference to flux ratio plot
    #colors = ["#1e1e1e", "#3399FF", "#F47174"]
    titles = ["All Matches", "Excluded Fake Matches", "Excluded Flagged Matches"]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    # print(xcenters1, mean_relSdiff_true)
    # print(len(xcenters1), len(mean_relSdiff_true))
    # If error, or weird plot (-1) it is due to empty bin --> skip plotting (the bars) these points
    step = 5
    for i in range(3):
        if i == 0:
            y = np.array([val for k, val in enumerate(mean_relSdiff_true) if (val != -1 and k%step == 0)])
            x = np.array([val for k, val in enumerate(xcenters1) if (mean_relSdiff_true[k] != -1 and k%step == 0)])
            xerr_bin = np.array([val for k, val in enumerate(np.diff(xedges1)) if (mean_relSdiff_true[k] != -1 and k%step == 0)])
            yerr_bin = np.array([abs((xcenters1[k]-val) - (ycenters1[k] + np.diff(yedges1)[k]))/((xcenters1[k]-val)) for k, val in enumerate(np.diff(xedges1)) if (mean_relSdiff_true[k] != -1 and k%step == 0)])
            yerr = np.array([val for k, val in enumerate(std_relSdiff_true) if (mean_relSdiff_true[k] != -1 and k%step == 0)])
            yerr_comb = np.array([val1 if val1 >= val2 else val2 for val1, val2 in zip(yerr, yerr_bin)])
            axs[i].errorbar(x, y, yerr=yerr_comb, xerr=xerr_bin, color="red", alpha=0.9, marker='o', lw=2, fillstyle='none', label="Upper error", capsize=3)
            # axs[i].errorbar(x, y, yerr=yerr_bin, xerr=xerr_bin, color="#3399FF", alpha=0.9, lw=2, label="Bin size", capsize=3, linestyle="None")
            axs[i].set_ylabel(r"$\left<\left|S_{in} - S_{out}\right|/S_{in}\right>$ (mJy/beam)")
        if i == 1:
            y = np.array([val for k, val in enumerate(mean_relSdiff_rnd) if (val != -1 and k%step == 0)])
            x = np.array([val for k, val in enumerate(xcenters1) if (mean_relSdiff_rnd[k] != -1 and k%step == 0)])
            xerr_bin = np.array([val for k, val in enumerate(np.diff(xedges1)) if (mean_relSdiff_rnd[k] != -1 and k%step == 0)])
            yerr_bin = np.array([abs((xcenters1[k]-val) - (ycenters1[k] + np.diff(yedges1)[k]))/((xcenters1[k]-val)) for k, val in enumerate(np.diff(xedges1)) if (mean_relSdiff_rnd[k] != -1 and k%step == 0)])
            yerr = np.array([val for k, val in enumerate(std_relSdiff_rnd) if (mean_relSdiff_rnd[k] != -1 and k%step == 0)])
            yerr_comb = np.array([val1 if val1 >= val2 else val2 for val1, val2 in zip(yerr, yerr_bin)])
            axs[i].errorbar(x,y, yerr=yerr_comb, xerr=xerr_bin, color="red", alpha=0.9, marker='o', lw=2, fillstyle='none', label="Upper error", capsize=3)
        if i == 2:
            y = np.array([val for k, val in enumerate(mean_relSdiff_flag) if (val != -1 and k%step == 0)])
            x = np.array([val for k, val in enumerate(xcenters1) if (mean_relSdiff_flag[k] != -1 and k%step == 0)])
            xerr_bin = np.array([val for k, val in enumerate(np.diff(xedges1)) if (mean_relSdiff_flag[k] != -1 and k%step == 0)])
            yerr_bin = np.array([abs((xcenters1[k]-val) - (ycenters1[k] + np.diff(yedges1)[k]))/((xcenters1[k]-val)) for k, val in enumerate(np.diff(xedges1)) if (mean_relSdiff_flag[k] != -1 and k%step == 0)])
            yerr = np.array([val for k, val in enumerate(std_relSdiff_flag) if (mean_relSdiff_flag[k] != -1 and k%step == 0)])
            yerr_comb = np.array([val1 if val1 >= val2 else val2 for val1, val2 in zip(yerr, yerr_bin)])
            axs[i].errorbar(x, y, yerr=yerr_comb, xerr=xerr_bin, color="red", alpha=0.9, marker='o', lw=2, fillstyle='none', label=r"$max(\sigma_{size_{bin}}, \sigma_{spread})$", capsize=3)
        axs[i].set_xlabel(r"$S_{in}$ (mJy/beam)")
        axs[i].set_title(titles[i], fontsize=12, alpha=0.9)
        axs[i].set_xscale('log')
        xlim = axs[i].get_xlim()
        axs[i].hlines(y=0, xmin=xlim[0], xmax=xlim[1], linestyle='dashed', color='black', alpha=0.9, linewidth=1)
        axs[i].legend()
        axs[i].set_ylim([-1, 2])

    fig.savefig(save2, dpi=350)
    plt.close('all')

def PS_plot_old(matches_df, save):
    fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharey=True)
    #fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharey=True)

    for i in range(2):
        Sdiff_all = np.array(matches_df['Sin_true'])*1000 - np.array(matches_df['Sout1'])*1000
        x_bins = np.linspace(-20, 20, 50)
        y_bins = np.linspace(0, 8, 50)
        #hist1, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, bins=(50,50))
        c1, xedges1, yedges1, img1 = axs[i,0].hist2d(Sdiff_all, np.array(matches_df['true_offset']), bins=(x_bins, y_bins), cmap='inferno')
        #hexplot1 = axs[0].hexbin(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, C=hist1, cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
        cbar1 = fig.colorbar(img1, ax=axs[i,0], label="Number of Matches")
        cbar1.ax.tick_params(labelsize=6)

        if i == 0:
            Sdiff_rnd = np.array(matches_df['Sin_rnd'])*1000 - np.array(matches_df['Sout2'])*1000
            #hist2, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_edges, y_edges))
            c2, xedges2, yedges2, img2 = axs[i,1].hist2d(Sdiff_rnd, matches_df['rnd_offset'], bins=(x_bins, y_bins), cmap='inferno')
            #hexplot2 = axs[1].hexbin(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, C=hist2, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
            cbar2 = fig.colorbar(img2, ax=axs[i,1], label="Number of Matches")
            cbar2.ax.tick_params(labelsize=6)
        elif i == 1:
            Sdiff_flag = np.array(matches_df['Sin_flag'])*1000 - np.array(matches_df['Sout3'])*1000
            #hist2, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_edges, y_edges))
            c2, xedges2, yedges2, img2 = axs[i,1].hist2d(Sdiff_flag, matches_df['flag_offset'], bins=(x_bins, y_bins), cmap='inferno')
            #hexplot2 = axs[1].hexbin(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, C=hist2, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
            cbar2 = fig.colorbar(img2, ax=axs[i,1], label="Number of Matches")
            cbar2.ax.tick_params(labelsize=6)
        c3 = c1 - c2
        #print(c3.shape, np.array(matches_df['Sin_true']).shape)
        #hexplot3 = axs[2].hexbin(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, C=hist3, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
        # c3, xedges, yedges, img3 = axs[2].hist2d(bins=(x_edges, y_edges), weights=c3.flatten(), cmap='inferno')
        pc = axs[i,2].pcolormesh(xedges2, yedges2, c3.T, cmap="inferno", vmin=0)
        cbar3 = fig.colorbar(pc, ax=axs[i,2], label="Number of Matches")
        cbar3.ax.tick_params(labelsize=6)

    for i in range(3):
        for j in range(2):
            #xlim = axs[j,i].get_xlim()
            #ylim = axs[j,i].get_ylim()
            #xtrue = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100, base=10)
            # Plot horizontal line equal to min detection threshold generated sources

            #axs[j, i].hlines(y=3.8, xmin=xlim[0], xmax=xlim[1], linestyle="dashed", color='#3CB043', linewidth=1)
            #ytrue = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 100, base=10)
            axs[j,i].set_ylabel(r"Offset ('')")
            axs[j,i].set_xlabel(r"$S_{in} - S_{out}$ (mJy/beam)")
            #axs[j,i].plot(xtrue, xtrue, color='red', alpha=0.7, label='True correlation', ls='dashed')

    fig.savefig(save, dpi=350)
    plt.close('all')

def offsetplot(matches_df, save):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    #fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharey=True)

    for i in range(2):
        Sdiff_all = np.array(matches_df['Sin_true'])*1000 - np.array(matches_df['Sout1'])*1000
        #hist1, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, bins=(50,50))
        img1 = axs[i,0].scatter(matches_df['true_xy_offset'][0], matches_df['true_xy_offset'][1], c=Sdiff_all, cmap='RdYlBu_r', s=1, vmin=-50, vmax=50)
        #hexplot1 = axs[0].hexbin(np.array(matches_df['Sin_true'])*1000, np.array(matches_df['Sout1'])*1000, C=hist1, cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
        cbar1 = fig.colorbar(img1, ax=axs[i,0])
        cbar1.set_label(label=r"$S_{in} - S_{out}$ (mJy/beam)", fontsize=10)
        if i == 0:
            Sdiff_rnd = np.array(matches_df['Sin_rnd'])*1000 - np.array(matches_df['Sout2'])*1000
            #hist2, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_edges, y_edges))
            img2 = axs[i,1].scatter(matches_df['rnd_xy_offset'][0], matches_df['rnd_xy_offset'][1], c=Sdiff_rnd, cmap='RdYlBu_r', s=1, vmin=-50, vmax=50)
            #hexplot2 = axs[1].hexbin(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, C=hist2, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
            cbar2 = fig.colorbar(img2, ax=axs[i,1])
            cbar2.set_label(label=r"$S_{in} - S_{out}$ (mJy/beam)", fontsize=10)
        elif i == 1:
            Sdiff_flag = np.array(matches_df['Sin_flag'])*1000 - np.array(matches_df['Sout3'])*1000
            #hist2, x_edges, y_edges = np.histogram2d(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, bins=(x_edges, y_edges))
            img2 = axs[i,1].scatter(matches_df['flag_xy_offset'][0], matches_df['flag_xy_offset'][1], c=Sdiff_flag, cmap='RdYlBu_r', s=1, vmin=-50, vmax=50)
            #hexplot2 = axs[1].hexbin(np.array(matches_df['Sin_rnd'])*1000, np.array(matches_df['Sout2'])*1000, C=hist2, gridsize=(50, 50), cmap='inferno', extent=(0, np.log10(150), 0, np.log10(175)), mincnt=1, xscale='log', yscale='log')
            cbar2 = fig.colorbar(img2, ax=axs[i,1])
            cbar2.set_label(label=r"$S_{in} - S_{out}$ (mJy/beam)", fontsize=10)


    for i in range(2):
        for j in range(2):
            #xlim = axs[j,i].get_xlim()
            #ylim = axs[j,i].get_ylim()
            #xtrue = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100, base=10)
            # Plot horizontal line equal to min detection threshold generated sources

            #axs[j, i].hlines(y=3.8, xmin=xlim[0], xmax=xlim[1], linestyle="dashed", color='#3CB043', linewidth=1)
            #ytrue = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), 100, base=10)
            #axs[j,i].plot(xtrue, xtrue, color='red', alpha=0.7, label='True correlation', ls='dashed')

            if j == 0:
                axs[i,j].set_ylabel(r"Offset $Y_{in} - Y_{out}$ ('')")
            if i == 0:
                if j == 0:
                    axs[i,j].set_title("""All Matches""", fontsize=12, alpha=.7)
                else:
                    axs[i,j].set_title("""All fake [upper]/flagged [bottom] Matches""", fontsize=12, alpha=.7)
            if i == 1:
                axs[i,j].set_xlabel(r"Offset $X_{in} - X_{out}$ ('')")
    fig.savefig(save, dpi=350)
    plt.close('all') 

def PSNR_COMP_plot(PSNR, labels, save):
    width = 0.2
    ind = np.arange(len(PSNR[0::2]))

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.barh(ind, PSNR[0::2], width, color='red', label=r'Best Validation Model', alpha=0.7)
    axs.barh(ind-width, PSNR[1::2], width, color='blue', label=r'Best Confusion Model', alpha=0.7)
    for i, v in enumerate(PSNR[0::2]):
        axs.text(v+0.15, i-0.03, str(np.round(v,decimals=2)), color='red', fontsize=8)
    for i, v in enumerate(PSNR[1::2]):
        axs.text(v+0.15, i-0.25, str(np.round(v,decimals=2)), color='blue', fontsize=8)
    axs.set(yticks=ind-width/2, yticklabels=labels, ylim=[2*width-1, len(labels)])
    axs.set_xlim([1, 75])
    axs.set_xscale('log')
    axs.set_xlabel("<PSNR> (dB)")
    axs.set_title("Average PSNR Model Comparison")
    axs.legend()
    axs.grid()
    #fig.tight_layout()
    fig.savefig(save, dpi=450)

def SkyPlot(ra1, dec1, ra2, dec2, labels, save):
    min_ra = np.min(ra1) if np.min(ra1) < np.min(ra2) else np.min(ra2)
    max_ra = np.max(ra1) if np.max(ra1) > np.max(ra2) else np.max(ra2)
    min_dec = np.min(dec1) if np.min(dec1) < np.min(dec2) else np.min(dec2)
    max_dec = np.max(dec1) if np.max(dec1) > np.max(dec2) else np.max(dec2)
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.scatter(ra1, dec1, s=1, color="blue", label=labels[0])
    axs.scatter(ra2, dec2, s=1, color="red", label=labels[1])
    axs.set_xlabel("RA (degrees)", fontsize=10)
    axs.set_ylabel("DEC (degrees)", fontsize=10)
    axs.set_title("Source Sky Projection Comparison")
    axs.set_xlim([min_ra-0.01, max_ra+0.01])
    axs.set_ylim([min_dec-0.01, max_dec+0.01])
    axs.legend()
    fig.savefig(save, dpi=450)



def confusion_COMP_plot(flux_bins, confusion_df_arr, labels, save_c, save_r):
    plt.clf()
    C_color = "#1e1e1e"
    
    # Completeness
    fig = plt.figure(figsize=(10, 5), frameon=True, facecolor='white')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey=ax1)

    axs = [ax1, ax2]
    
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins])
    for idx, confusion_df in enumerate(confusion_df_arr[0::2]):
        ax1.plot(flux*1000, confusion_df['C'], alpha=0.9, label=labels[idx], lw=1)
    for idx, confusion_df in enumerate(confusion_df_arr[1::2]):
        ax2.plot(flux*1000, confusion_df['C'], alpha=0.9, label=labels[idx], lw=1)


    for ax in axs:
        ax.set_xlabel(r"Herschel SPIRE $500\mu m$ Source Flux (mJy/beam)")
    
        # Plot Styling
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_xscale('log')

        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(axis='y', length=0)
        for tick_label in ax.yaxis.get_ticklabels():
            tick_label.set_fontsize(11)
        for tick_label in ax.xaxis.get_ticklabels():
            tick_label.set_fontsize(11)
            tick_label.set_horizontalalignment('center')
        ax.tick_params(direction='out', axis='x', length=7, color='0.1')
        ax.legend(loc='best')

    fig.suptitle(u'Completeness: Accuracy', color=C_color, fontstyle='italic')
    ax1.set_title(u'Best Validation Model', color=C_color, fontstyle='italic')
    ax2.set_title(u'Best Confusion Model', color=C_color, fontstyle='italic')

    ## Save the plot
    fig.savefig(save_c, dpi=350, edgecolor='white', facecolor='white')


    # Reliability
    fig = plt.figure(figsize=(10, 5), frameon=True, facecolor='white')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey=ax1)

    axs = [ax1, ax2]
    
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color='0.65', ls='-', lw=1.5, zorder=0)

    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins])
    for idx, confusion_df in enumerate(confusion_df_arr[0::2]):
        ax1.plot(flux*1000, confusion_df['R'], alpha=0.9, label=labels[idx], lw=1)
    for idx, confusion_df in enumerate(confusion_df_arr[1::2]):
        ax2.plot(flux*1000, confusion_df['R'], alpha=0.9, label=labels[idx], lw=1)


    for ax in axs:
        ax.set_xlabel(r"Herschel SPIRE $500\mu m$ Source Flux (mJy/beam)")
    
        # Plot Styling
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_xscale('log')

        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(axis='y', length=0)
        for tick_label in ax.yaxis.get_ticklabels():
            tick_label.set_fontsize(11)
        for tick_label in ax.xaxis.get_ticklabels():
            tick_label.set_fontsize(11)
            tick_label.set_horizontalalignment('center')
        ax.tick_params(direction='out', axis='x', length=7, color='0.1')
        ax.legend(loc='best')

    # Display analogy completeness and reliability
    ax1.set_title(u'Reliability: Precision (Best Validation Model)', color=C_color, fontstyle='italic')
    ax2.set_title(u'Reliability: Precision (Best Confusion Model)', color=C_color, fontstyle='italic')

    # Save the plot
    fig.savefig(save_r, dpi=350, edgecolor='white', facecolor='white')

def FluxReproduction_COMP_plot(bins, mean_relSdiff, MAD_relSdiff, labels, save_base):
    # 1 row, 2 col plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, title in enumerate(["Best Valid Model", "Best Confusion Model"]):
        for modelID, lbl in enumerate(labels):
            if i == 0:
                axs[0].plot(bins, mean_relSdiff[2*modelID + i], label=lbl)
                axs[1].plot(bins, MAD_relSdiff[2*modelID + i], label=lbl)
            elif i == 1:
                axs2[0].plot(bins, mean_relSdiff[2*modelID + i], label=lbl)
                axs2[1].plot(bins, MAD_relSdiff[2*modelID + i], label=lbl)
        axs[i].set_xlabel(r"Source Flux (mJy)")
        axs2[i].set_xlabel(r"Source Flux (mJy)")

        axs[i].legend(loc='best')
        axs2[i].legend(loc='best')

    axs[0].set_ylabel(r"$\left<\left|S_{in} - S_{out}\right|/S_{in}\right>$ (mJy/beam)")
    axs2[0].set_ylabel(r"$\left<\left|S_{in} - S_{out}\right|/S_{in}\right>$ (mJy/beam)")

    axs[1].set_ylabel(r"Average Relative Spread")
    axs2[1].set_ylabel(r"Average Relative Spread")

    fig.suptitle("Best Valid Model", color="#1e1e1e", fontstyle='italic')
    fig2.suptitle("Best Confusion Model", color="#1e1e1e", fontstyle='italic')
    
    fig.savefig(save_base + "_BestValid.pdf", dpi=350, edgecolor='white', facecolor='white')
    fig2.savefig(save_base + "_BestConfusion.pdf", dpi=350, edgecolor='white', facecolor='white')

def Correlation_COMP_plot(bins, corr_target, corr_err_target, corr_arr, corr_err_arr, labels, save):
    delta_corr = np.array([corr_target - corr_model for corr_model in corr_arr])
    delta_corr_err = np.array([np.sqrt(corr_err_target**2 + corr_err_model**2) for corr_err_model in corr_err_arr])

    # 1 row, 2 col plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    for i, title in enumerate(["Best Valid Model", "Best Confusion Model"]):
        for modelID, lbl in enumerate(labels):
            axs[i].plot(bins*3600, delta_corr[2*modelID + i], label=lbl)
            axs[i].fill_between(bins*3600, delta_corr[2*modelID + i] - delta_corr_err[2*modelID + i], delta_corr[2*modelID + i] + delta_corr_err[2*modelID + i], alpha=0.2)
        
        axs[i].legend(loc='best')
        axs[i].set_xlabel(r"$\theta$ (arcseconds)")
        axs[i].set_title(title, color="#1e1e1e", fontstyle='italic')

    axs[0].set_ylabel(r"$\Delta \omega(\theta)$")
    fig.savefig(save, dpi=350, edgecolor='white', facecolor='white')

    
def CorrelationPlotCheck(mean_target, std_target, mean_rnd, std_rnd, center_bins, save):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.plot(center_bins*3600, mean_rnd, color='blue', label=r'Random Field')
    axs.plot(center_bins*3600, mean_target, color='red', label=r'Clustered Field')
    axs.fill_between(center_bins*3600, mean_rnd-std_rnd, mean_rnd+std_rnd, alpha=0.3, color='blue')
    axs.fill_between(center_bins*3600, mean_target-std_target, mean_target+std_target, alpha=0.3, color='red')

    axs.legend()
    axs.set_xlabel(r"$\theta$ (arcseconds)")
    axs.set_ylabel(r"$\omega(\theta)$")
    axs.set_title("Random Field Check")
    fig.savefig(save, dpi=450)


def CorrelationPlot(mean_target, std_target, mean_gen, std_gen, center_bins, save):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.plot(center_bins*3600, mean_target, color='blue', label=r'Target')
    axs.plot(center_bins*3600, mean_gen, color='red', label=r'Generated')
    axs.fill_between(center_bins*3600, mean_target-std_target, mean_target+std_target, alpha=0.3, color='blue')
    axs.fill_between(center_bins*3600, mean_gen-std_gen, mean_gen+std_gen, alpha=0.3, color='red')
    axs.legend()
    axs.set_xlabel(r"$\theta$ (arcseconds)")
    axs.set_ylabel(r"$\omega(\theta)$")
    fig.savefig(save, dpi=450)

def insetplot_old(gen, target, Y_cat, save):
    gen = np.squeeze(gen)
    target = np.squeeze(target)

    ## estimate noise level
    _, __, std_noise = sigma_clipped_stats(target, sigma=2, maxiters=50)
    #print(np.std(target.flatten()[target.flatten() <= 0]))

    # The aperture flux hexplot
    fig = plt.figure(figsize=(11,11))
    gs = GridSpec(2, 2 , height_ratios=[1.5, 1], hspace=0.005)
    
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])

    ax1.imshow(gen, origin='lower', cmap='gnuplot2', vmin=0, vmax=np.max(target))
    ax2.imshow(target, origin='lower', cmap='gnuplot2',  vmin=0, vmax=np.max(target))

    ax1.set_title("Generated Image", fontsize=13, weight='bold', alpha=.8)
    ax2.set_title("True Image", fontsize=13, weight='bold', alpha=.8)

    # Make a zoom
    y0 = 100
    ysize = 40
    x0 = 60
    xsize = 40
    axins1 = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])


    #sources = Y_cat[np.logical_and(y0 < Y_cat[:,1] < y0+ysize, x0 < Y_cat[:,0] < x0+xsize)]
    sources = Y_cat[np.where((Y_cat[:,2] < y0+ysize) & (Y_cat[:,1] < x0+xsize))]
    sources = sources[np.where((sources[:,2] > y0) & (sources[:,1] > x0))][:,0:-1]
    try:
        sprof_idx = np.argmax(sources[:,0])
    except:
        plt.close('all')
        return

    if len(sources) == 0:
        return

    source_finder = DAOStarFinder(fwhm=7.9, threshold=3*std_noise)
    gen_sources = source_finder(gen)
    try:
        tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
    except:
        plt.close('all')
        return
    tr_gen = tr_gen[np.where((tr_gen[:,-1] < y0+ysize) & (tr_gen[:,-2] < x0+xsize))]
    tr_gen = tr_gen[np.where((tr_gen[:,-1] > y0) & (tr_gen[:,-2] > x0))]
    if len(tr_gen) == 0:
        plt.close('all')
        return
    # sources[:,0] -= xsize
    # sources[:,1] -= ysize

    axins1.imshow(gen, origin='lower', cmap='gnuplot2', vmin=0, vmax=np.max(target))
    axins2.imshow(target, origin='lower', cmap='gnuplot2', vmin=0, vmax=np.max(target))

    Flag = False
    for idx, s in enumerate(sources):
        if idx == sprof_idx:
            axins1.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='green', linewidths=1, facecolors='none')

            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] > 8:
                plt.close('all')
                return
        else:
            axins1.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')
            axins2.scatter(s[1], s[2], marker="o", s=s[0]*10000, color='red', linewidths=1, facecolors='none')

    for idx, s in enumerate(tr_gen):
        peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        if idx == rmin_idx:
            r_flag = np.sqrt((s[0] - sources[:,1])**2 + (s[1] - sources[:,2])**2)
            # Reliability Issue Detection
            potential_flags = np.where(r_flag <= 8)[0]
            if len(potential_flags) > 1:
                true_sources_in_fhwm = sources[:,1:][potential_flags]
                # Try to connect each true source under the PSF 'umbrella' of a generated source
                # If this is possible, we have no issue and the generated source is "fine"
                # If this is not possible, there is a source-source seperation > PSF. 
                # These should not be under the PSF of the generated source, and we should have atleast 2 seperate generated sources.
                if not can_connect_circles(true_sources_in_fhwm, 8):
                    # Flag found!
                    Flag = True
            axins1.scatter(s[0], s[1], marker="s", s=peak*10000, color='green', linewidths=1, facecolors='none')
            sgenprof_peak = gen_sources['peak'][np.where((gen_sources['xcentroid'] == s[0]) & (gen_sources['ycentroid'] == s[1]))]
        else:
            axins1.scatter(s[0], s[1], marker="s", s=peak*10000, color='red', linewidths=1, facecolors='none')

    axins1.set_xlim(x0, x0+xsize)
    axins1.set_ylim(y0, y0+ysize)
    axins2.set_xlim(x0, x0+xsize)
    axins2.set_ylim(y0, y0+ysize)
    axins1.set_xticklabels('')
    axins1.set_yticklabels('')
    axins2.set_xticklabels('')
    axins2.set_yticklabels('')

    if Flag:
        axins1.spines['bottom'].set_color('green')
        axins1.spines['top'].set_color('green') 
        axins1.spines['right'].set_color('green')
        axins1.spines['left'].set_color('green')

        axins2.spines['bottom'].set_color('green')
        axins2.spines['top'].set_color('green')
        axins2.spines['right'].set_color('green')
        axins2.spines['left'].set_color('green')
    else:
        axins1.spines['bottom'].set_color('gray')
        axins1.spines['top'].set_color('gray') 
        axins1.spines['right'].set_color('gray')
        axins1.spines['left'].set_color('gray')

        axins2.spines['bottom'].set_color('gray')
        axins2.spines['top'].set_color('gray')
        axins2.spines['right'].set_color('gray')
        axins2.spines['left'].set_color('gray')

    ax1.indicate_inset_zoom(axins1, edgecolor='gray')
    ax2.indicate_inset_zoom(axins2, edgecolor='gray')


    # Profile flux of a source
    strueprof_x = sources[sprof_idx][1]
    strueprof_y = sources[sprof_idx][2]
    strueprof_peak = sources[sprof_idx][0]

    if std_noise > strueprof_peak or  std_noise > sgenprof_peak:
        plt.close('all')
        return

    sgenprof_x = tr_gen[rmin_idx][0]
    sgenprof_y = tr_gen[rmin_idx][1]
    
    h_slicetrue = np.arange(int(np.round(strueprof_x, decimals=0)) - 10, int(np.round(strueprof_x, decimals=0)) + 10, 1)
    h_slicegen = np.arange(int(np.round(sgenprof_x, decimals=0)) - 10, int(np.round(sgenprof_x, decimals=0)) + 10, 1)

    ax3 = plt.subplot(gs[1,:])
    ax3.plot((h_slicegen - int(sgenprof_x))*1, gen[int(sgenprof_y), h_slicegen]*1000, color='green', alpha=0.7, ls='dashed', marker='o', fillstyle='none', label="Generated Profile")
    ax3.plot((h_slicetrue - int(strueprof_x))*1, target[int(strueprof_y), h_slicetrue]*1000, color='blue', alpha=0.9, ls='-', marker='o', fillstyle='none', label="True Profile")
    ax3.yaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax3.xaxis.grid(True, color='0.65', ls='-', lw=1., zorder=0)
    ax3.hlines(y=std_noise*1000, xmin=(h_slicetrue[0]- int(strueprof_x))*1, xmax=(h_slicetrue[-1]- int(strueprof_x))*1, color='black', ls='dotted', label=rf'Noise $\sigma={std_noise*1000:.1f}$ mJy')
    #ax3.vlines(x = sprof_x, ymin=np.min(target[int(np.round(sprof_y)), h_slice]), ymax=np.max(target[int(np.round(sprof_y)), h_slice]), color='black', ls='dotted')
    ax3.scatter(0, strueprof_peak*1000, marker='x', color='blue', label='True Peak Flux', s=15)
    ax3.scatter(0, sgenprof_peak*1000, marker='x', color='green', label='Generated Peak Flux', s=15)
    ax3.set_xlabel("Offset ('')")
    ax3.set_ylabel(r'$S$ (mJy\beam)')

    ax3.set_xlim([(h_slicetrue[0]- int(strueprof_x))*1, (h_slicetrue[-1]- int(strueprof_x))*1])

    ax3.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper right')
    if Flag:
        ax1.text(x=0.05, y=0.84, s="""FLAGGED""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure, color="green")

    ax1.text(x=0.05, y=0.88, s=rf"""Generated Source detection threshold is set at $3\sigma={1000*std_noise*3:.1f}mJy/beam$""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    ax1.text(x=0.05, y=0.86, s=rf"""Target Source detection threshold is set at $5\sigma={1000*std_noise*5:.1f}mJy/beam$""", ha='left', fontsize=10, alpha=.7, transform=fig.transFigure)
    ax1.text(x=0.05, y=0.9, s="Comparison Plot", ha='left', fontsize=13, weight='bold', alpha=.8, transform=fig.transFigure)
    plt.savefig(save, dpi=350)
    if Flag:
        print("flag!")
    plt.close('all')



# def offsetplot(target_galx_aperfluxes, gen_galx_aperfluxes, distances, x_offset, y_offset, save1, save2, save3):
#     # SCATTERPLOT
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(1,1,1)
#     # Standard deviation of the distances
#     std_dev = np.std(distances)
#     mean_dist = np.mean(distances)
#     median_dist = np.median(distances)

#     # Difference in aperture flux
#     aper_diff = np.array(target_galx_aperfluxes) - np.array(gen_galx_aperfluxes)

#     # Scatter plot
#     ax.scatter(aper_diff*1000, distances, s=1, color='black')
#     ax.set_xlim([-20, 20])

#     # Add labels and title
#     ax.set_xlabel(r"$S_{aper, in} - S_{aper, out} \ [mJy/pixel]$", fontsize=13)
#     ax.set_ylabel(r"$Offset ('')$", fontsize=13)
#     ax.set_title("Offset Plot", fontsize=16, weight='bold', alpha=.8)

#     # Save figure
#     plt.savefig(save1, dpi=350)
#     plt.close('all')

#     # SCATTERPLOT xpos, ypos offset
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(1,1,1)

#     # Scatter plot
#     ax.scatter(x_offset, y_offset, s=1, color='black')

#     # Add labels and title
#     ax.set_xlabel(r"$x_{true} - x_{detected}$ ('')", fontsize=13)
#     ax.set_ylabel(r"$y_{true} - y_{detected}$ ('')", fontsize=13)
#     ax.set_title("Radial-Components Offset Plot", fontsize=16, weight='bold', alpha=.8)

#     # Save figure
#     plt.savefig(save2, dpi=350)
#     plt.close('all')
    
#     # HISTOGRAMPLOT
#     fig = plt.figure(figsize=(7,5))
#     ax = fig.add_subplot(1,1,1)
#     # Standard deviation of the distances
#     y, x, _ = ax.hist(distances, edgecolor='black', linewidth=1, bins=50)
#     ax.vlines(x=mean_dist, ymin=0, ymax=max(y), linestyles='dashed', color='red', label=f"mean: {mean_dist:.1f}('')")
#     ax.vlines(x=median_dist, ymin=0, ymax=max(y), linestyles='dashed', color='green', label=f"median: {median_dist:.1f}('')")

#     # Add labels and title
#     ax.set_xlabel(r"$Offset ('')$", fontsize=13)
#     ax.set_ylabel("Frequency", fontsize=13)
#     ax.set_title("Histogram Plot", fontsize=16, weight='bold', alpha=.8)
#     ax.legend(loc='best')
#     ax.set_ylim([0, max(y)])


#     # Save figure
#     plt.savefig(save3, dpi=350)
#     plt.close('all')


def TestMetricPlot(Lh, Lstats, Lflux, avg_r, avg_c, distances, SSIM, save):
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    # Stacked barplot, Loss metrics
    x = ['']
    y = np.array([Lh, Lstats, Lflux])
    y_lbl = ['Lh', 'Lstats', 'Lflux']
    color = ['b', 'y', 'g']
    for i in range(len(y)):
        if i == 0:
            ax1.bar(x, y[i], color=color[i], label=f'{y_lbl[i]} = {y[i]:.2f}')
        else:
            ax1.bar(x, y[i], color=color[i], bottom=np.sum(y[:i]), label=f'{y_lbl[i]} = {y[i]:.2f}')
    ax1.set_ylabel("Loss value")
    ax1.legend()
    ax1.set_title("Unparametrized loss metric")
    
    # Barplot confusion metric
    x = ['Avg_r', 'Avg_c']
    y = [avg_r, avg_c]
    ax2.bar(x, y, color='blue')
    for i in range(2):
        ax2.text(i, y[i], f"{y[i]:.2f}", ha = 'center')
    ax2.set_title("Confusion metric")
    ax2.set_ylim([0, 1.])
    ax2.set_ylabel("Ratio")

    # Barplot offset metric
    x = ['Median', 'Mean']
    y = [np.median(distances), np.mean(distances)]
    ax3.bar(x, y, color='blue')
    for i in range(2):
        ax3.text(i, y[i], f"{y[i]:.1f}", ha = 'center')
    #ax3.legend([f'Median offset = {np.median(distances):.1f}', f'Mean offset = {np.mean(distances):.1f}'])
    ax3.set_title("Offset metric")
    ax3.set_ylabel("Offset ('')")

    # Barplot SSIM metric
    x = ['']
    ax4.bar(x, [SSIM], color='blue')
    ax4.legend([f'SSIM = {SSIM:.2f}'])
    ax4.set_title("SSIM metric")
    ax4.set_ylabel("Correlation Coefficient")

    fig.savefig(save, dpi=350)


def SSIM_plot(SSIM_img, save):
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1,1,1)

    img = ax1.imshow(SSIM_img, vmin=0.4, vmax=1, aspect='auto', cmap='viridis')
    fig.colorbar(img)
    fig.savefig(save, dpi=350)


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
        f.write("\nReliability FLAG Values\n")
        for lbl, val in zip(confusion_df['Flux bins'], confusion_df['flag_R']):
            f.write(f"{lbl}: {val}\n")


