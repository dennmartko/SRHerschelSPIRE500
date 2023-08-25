import os
import numpy as np
from matplotlib import pyplot as plt

def PSNR_plot(epochs_l, PSNR_l, labels= [""], save_path = ""):
    # Create a new figure
    plt.figure(figsize=(8,8))

    # Plot each line
    for idx, PSNR in enumerate(PSNR_l):
        plt.plot(epochs_l[idx], PSNR, label=labels[idx])


    # Customize labels and ticks
    plt.xlabel("Epoch Number")
    plt.ylabel("<PSNR> (dB)")

    plt.legend()
    plt.savefig(os.path.join(save_path))
    # Show the plot
    plt.close()

def CR_plot(epochs_l, C_l, R_l, labels= [""], save_path = ""):
    # Create a new figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    for idx in range(len(C_l)):
        axs[0].plot(epochs_l[idx], C_l[idx], label=labels[idx])
        axs[1].plot(epochs_l[idx], R_l[idx], label=labels[idx])

    # Customize labels and ticks
    axs[0].set_xlabel("Epoch Number")
    axs[1].set_xlabel("Epoch Number")
    axs[0].set_ylabel("Completeness (%)")
    axs[1].set_ylabel("Completeness (%)")
    # Add legends to both subplots
    axs[0].legend()
    axs[1].legend()

    plt.savefig(os.path.join(save_path))
    # Show the plot
    plt.close()

def FitParams_plot(epochs_l, d_a_l, d_b_l, mape_l, labels= [""], save_path = ""):
    # Create a new figure
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=False)

    for idx in range(len(d_a_l)):
        axs[0].plot(epochs_l[idx], d_a_l[idx], label=labels[idx])
        axs[1].plot(epochs_l[idx], d_b_l[idx], label=labels[idx])
        axs[2].plot(epochs_l[idx], mape_l[idx], label=labels[idx])

    # Customize labels and ticks
    axs[0].set_xlabel("Epoch Number")
    axs[1].set_xlabel("Epoch Number")
    axs[2].set_xlabel("Epoch Number")
    axs[0].set_ylabel(r"$|1. - a|$")
    axs[1].set_ylabel(r"$|b|$")
    axs[2].set_ylabel(r"MAPE (%)")

    # Add legends to both subplots
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.savefig(os.path.join(save_path))
    # Show the plot
    plt.close()