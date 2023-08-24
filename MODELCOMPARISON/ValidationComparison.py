import os
import configparser
import datetime

import numpy as np
import pandas as pd

from PlotLib.validationplots import *

path_to_config = "ValidationComparison.ini"

config = configparser.ConfigParser()
config.read(path_to_config)

print(f"{datetime.datetime.now()} - Reading Configuration File...")
path_to_model_libdir = config["COMMON"]['model_libdir'].rstrip().lstrip()
path_to_figures = config["COMMON"]['figures_output_dir'].rstrip().lstrip()
comparison_name = config["COMMON"]['comparison_name'].rstrip().lstrip()
outdir = os.path.join(path_to_figures, comparison_name)

if not os.path.isdir(outdir):
    os.mkdir(outdir)

models = [i.strip(' ') for i in config["MODELS"]['models'].rstrip().lstrip().split(",")]

vallosshist_paths = [os.path.join(os.path.join(path_to_model_libdir, model), 'ValidationLossHistory.npz') for model in models]
valloshist_l = [np.load(vallosshist_path) for vallosshist_path in vallosshist_paths]
print(f"{datetime.datetime.now()} - Config Read!")


print(f"{datetime.datetime.now()} - Creating Validation Comparison Plots...")
PSNR_plot([valloshist_l[idx]["epochs"] for idx in range(len(models))], [valloshist_l[idx]["PSNR"] for idx in range(len(models))], labels=models, save_path=os.path.join(outdir, "PSNR_comparison.pdf"))
CR_plot([valloshist_l[idx]["epochs"] for idx in range(len(models))], [valloshist_l[idx]["Cglobal"] for idx in range(len(models))], [valloshist_l[idx]["Rglobal"] for idx in range(len(models))], labels=models, save_path=os.path.join(outdir, "ReliabilityCompleteness_comparison.pdf"))
FitParams_plot([valloshist_l[idx]["epochs"] for idx in range(len(models))], [valloshist_l[idx]["d_a"] for idx in range(len(models))], [valloshist_l[idx]["d_b"] for idx in range(len(models))], [valloshist_l[idx]["mape"] for idx in range(len(models))], labels=models, save_path=os.path.join(outdir, "FitParams_comparison.pdf"))
print(f"{datetime.datetime.now()} - Completed!")


