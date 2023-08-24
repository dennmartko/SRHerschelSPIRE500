import itertools
import gc
import os
import configparser

import tensorflow as tf
import numpy as np

#from train_functions import SRTrainer
from train_functions_gpu import SRTrainerGPU

from tqdm import tqdm

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("TrainingConfig.ini")
    # Toggle GPU/CPU
    if config['COMMON']['use_gpu'].rstrip().lstrip() == "False":
        # TODO FIX CPU VERSION!
        print("CPU Version broken!")
    else:
        SRModelTrain = SRTrainerGPU("TrainingConfig.ini")
        SRModelTrain.LoadTrainingData()
        SRModelTrain.BuildModel()
        SRModelTrain.Train()
        SRModelTrain.TrainingAnalysis()
