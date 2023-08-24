######################
###     IMPORTS    ###
######################

import tensorflow as tf
tf.random.set_seed(42)

import os
import random
import configparser
import datetime
import gc
import imageio
import glob
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from ModelArchitectures.PaperModel import Generator, Discriminator
from loss_functions import non_adversarial_loss
from PlotLib.PlotFunctionsTrain import TrainingSnapShot, plot_line_chart, TrainingSnapShotFeatureSample
from metric_funcs import confusion_score, find_matches, compute_PSNR_batch, fill_pixcatalog, get_matches_n_confusion, fit_flux_distribution
from scipy.optimize import curve_fit

random.seed(10)


#######################
###   TRAIN CLASS   ###
#######################
class SRTrainerGPU:
    def __init__(self, path_to_config, idx=None) -> None:
        # Load configuration file for training
        self.config = configparser.ConfigParser()
        self.config.read(path_to_config)

        # Create any missing directories and initialize necessary parameters
        self.DIM = (424, 424)

        # Paths with train data
        self.path_train = self.config['COMMON']['path_train'].strip()

        # Indicate purpose of run
        self.RUN_NAME = self.config['COMMON']['RUN_NAME'].strip()

        # Indicate whether we continue training of the model, or its trained for the first time
        self.first_run = self.config['COMMON']['first_run'].strip() == "True"

        # Create folder with trained models
        self.models_lib_path = self.config['COMMON']['model_outdir'].strip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)

        # if verbose is TRUE create file TrainingLog.log in model_path
        self.verbose = self.config['COMMON']['training_verbose'].strip() == "True"
        self.logfile_path = os.path.join(self.model_path, "TrainingLog.log")

        os.makedirs(self.models_lib_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        if self.verbose and self.first_run:
            open(self.logfile_path, 'w').close()

        # Set classes
        classes_input = [i.strip() for i in self.config['COMMON']['input'].split(",")]
        class_target = self.config['COMMON']['target'].strip()
        self.classes = classes_input + [class_target]  # Input classes first, target class last

        # Total samples
        self.TOTAL_SAMPLES = len(os.listdir(os.path.join(self.path_train, self.classes[0])))#int(np.round(len(os.listdir(os.path.join(self.path_train, self.classes[0])))*0.2))
        tdir_out = os.path.join(self.model_path, "train_results")
        self.tdir_out_progress = os.path.join(tdir_out, "train_progress")
        self.tdir_out_analysis = os.path.join(tdir_out, "train_analysis")

        os.makedirs(tdir_out, exist_ok=True)
        os.makedirs(self.tdir_out_progress, exist_ok=True)
        os.makedirs(self.tdir_out_analysis, exist_ok=True)

        if self.verbose:
            self.printlog(f"{datetime.datetime.now()} - Successfully Configured Training Settings!")

            # List device used: GPU/CPU
            self.printlog(f"{datetime.datetime.now()} - GPU assigned!")

    def printlog(self, message):
        if self.verbose:
            with open(self.logfile_path, 'a') as log_file:
                log_file.write(message + "\n")

    def LoadTrainingData(self):
        # Write to log
        if self.verbose == True:
            self.printlog(f"{datetime.datetime.now()} - Call to Load Data....")

        # Split Training data into validation and training set according to parameter validation_ratio
        validation_ratio = float(self.config['TRAINING PARAMETERS']['validation_ratio'].rstrip().lstrip())
        indices = np.arange(0, self.TOTAL_SAMPLES)
        TOTAL_SAMPLES_VALID = round(validation_ratio * self.TOTAL_SAMPLES)
        n_valid_indx = np.array(random.sample(indices.tolist(), TOTAL_SAMPLES_VALID))

        ## Load training and validation data into memory
        self.train_arr_X = np.zeros((self.TOTAL_SAMPLES - TOTAL_SAMPLES_VALID, 4, 106, 106))
        self.train_arr_Y = np.zeros((self.TOTAL_SAMPLES - TOTAL_SAMPLES_VALID, 1, 424, 424))
        self.valid_arr_X = np.zeros((TOTAL_SAMPLES_VALID, 4, 106, 106))
        self.valid_arr_Y = np.zeros((TOTAL_SAMPLES_VALID, 1, 424, 424))

        valid_idx = 0
        train_idx = 0

        for i in range(self.TOTAL_SAMPLES):
            if i not in n_valid_indx:
                for k in range(len(self.classes)):
                    with fits.open(os.path.join(self.path_train, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits"), memmap=False) as hdu:
                        if k == len(self.classes) - 1:
                            self.train_arr_Y[train_idx] = hdu[0].data
                            arr = np.array([list(row) for row in hdu[1].data])
                            if train_idx == 0:
                                arr = np.column_stack((arr, np.full(len(arr), train_idx)))
                                self.target_image_sources_cat_train = arr.copy()
                            else:
                                arr = np.column_stack((arr, np.full(len(arr), train_idx)))
                                self.target_image_sources_cat_train = np.vstack((self.target_image_sources_cat_train, arr))

                            del arr;
                        else:
                            self.train_arr_X[train_idx][k] = hdu[0].data

                train_idx += 1;
            elif i in n_valid_indx:
                for k in range(len(self.classes)):
                    with fits.open(os.path.join(self.path_train, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits"), memmap=False) as hdu:
                        if k == len(self.classes) - 1:
                            self.valid_arr_Y[valid_idx] = hdu[0].data
                            arr = np.array([list(row) for row in hdu[1].data])
                            if valid_idx == 0:
                                arr = np.column_stack((arr, np.full(len(arr), valid_idx)))
                                self.target_image_sources_cat_valid = arr.copy()
                            else:
                                arr = np.column_stack((arr, np.full(len(arr), valid_idx)))
                                self.target_image_sources_cat_valid = np.vstack((self.target_image_sources_cat_valid, arr))

                            del arr;
                        else:
                            self.valid_arr_X[valid_idx][k] = hdu[0].data
                valid_idx += 1;

        self.train_arr_X = self.train_arr_X.astype(np.float32)
        self.train_arr_Y = self.train_arr_Y.astype(np.float32)
        self.valid_arr_X = self.valid_arr_X.astype(np.float32)
        self.valid_arr_Y = self.valid_arr_Y.astype(np.float32)
        #print(self.target_image_sources_cat_valid[self.target_image_sources_cat_valid[:,-2] < 0])
        # Write to log
        if self.verbose == True:
            self.printlog(f"{datetime.datetime.now()} - Data Loaded in memory!")
            self.printlog(f"{datetime.datetime.now()} - Training Samples: (Xshape, Yshape) = {(self.train_arr_X.shape, self.train_arr_Y.shape)}, Validation Samples: (Xshape, Yshape) = {(self.valid_arr_X.shape, self.valid_arr_Y.shape)}")
        
        self.dtypes = {"peak": np.float32, "xpix": np.float32, "ypix":np.float32, "ImageID":np.int32}
        self.cat_cols = ["peak", "xpix", "ypix", "ImageID"]
        self.train_catalog = pd.DataFrame(data=self.target_image_sources_cat_train, columns=self.cat_cols).astype(self.dtypes)
        self.valid_catalog = pd.DataFrame(data=self.target_image_sources_cat_valid, columns=self.cat_cols).astype(self.dtypes)
        del self.target_image_sources_cat_train;
        del self.target_image_sources_cat_valid;
        # Free memory
        gc.collect()

    def get_real_images(self):
        # NCHW --> GPU
        # NHWC --> CPU
        #Y = np.empty(shape=(self.BATCH_SIZE, 1, self.DIM[0], self.DIM[1]), dtype="float32")
        draw = random.sample(range(0, self.train_arr_X.shape[0]), self.BATCH_SIZE)
        draw = sorted(draw) 

        X = self.train_arr_X[draw]
        Y = self.train_arr_Y[draw]

        # mask_tensor = tf.zeros(self.train_arr_X.shape[0], dtype=tf.int32)
        # mask_tensor = tf.tensor_scatter_nd_update(mask_tensor, tf.expand_dims(draw, axis=1), tf.ones_like(draw, dtype=tf.int32))
        return X, Y

    def BuildModel(self):
        if self.first_run:
            # Build the model given the chosen architecture
            self.generator = Generator((4, 106, 106), "channels_first", 32, 4, (2, 4, 8, 16)) #1.5, 2.5, 4, 6, 16
            self.discriminator = Discriminator((1, 424, 424), (4, 106, 106),  "channels_first", 32)
            if self.verbose == True:
                self.printlog(f"{datetime.datetime.now()} - Model Architecture Loaded!")
                self.generator.summary(print_fn=lambda x: self.printlog(x))
                self.discriminator.summary(print_fn=lambda x: self.printlog(x))
        else:
            self.generator = tf.keras.models.load_model(os.path.join(self.model_path, 'checkpoint_GModel'))
            self.discriminator = tf.keras.models.load_model(os.path.join(self.model_path, 'checkpoint_DModel'))
            if self.verbose == True:
                self.printlog(f"{datetime.datetime.now()} - Previous Checkpoint Models Loaded!")

    def evaluate_metrics(self):
        max_bin_value = 125 # mJy
        std_noise = 2. # mJy
        min_bin_value = std_noise #mJy
        num_bins = 15
        rnd_its = 4
        batch_size = 32
        values = np.logspace(np.log10(min_bin_value), np.log10(max_bin_value), num_bins + 1, base=10)/1000
        ## Zip the values array with itself shifted by one position to the left to create tuples of the left and right bounds of each bin
        self.flux_bins = list(zip(values[:-1], values[1:]))
        zero_list = np.zeros(len(self.flux_bins))

        confusion_dict = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
        confusion_df = pd.DataFrame(confusion_dict)

        reconstructed_catalog = {"peak": [], "xpix": [], "ypix": [], "ImageID": []}

        coincidence_matching_args = {"max_offset_pixels": 30, "max_distance": 7.9, "ReproductionRatio_min": 0.5, "ReproductionRatio_max": 2.5}
        matching_args = {k: v for k, v in coincidence_matching_args.items() if k != "max_offset_pixels"}

        its = self.valid_arr_X.shape[0]//batch_size

        PSNR = 0
        for batch_idx in range(its):
            # Load the batch of data
            X = self.valid_arr_X[batch_idx*batch_size:batch_size*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.valid_arr_X[batch_idx*batch_size:].astype(np.float32)
            Y = self.valid_arr_Y[batch_idx*batch_size:batch_size*(batch_idx + 1)].astype(np.float32) if batch_idx != (its - 1) else self.valid_arr_Y[batch_idx*batch_size:].astype(np.float32)
           
            # Compute the corresponding ImageIDs
            idx_arr = np.arange(batch_idx*batch_size, batch_size*(batch_idx + 1)) if batch_idx != (its - 1) else np.arange(batch_idx*batch_size, self.valid_arr_X.shape[0])
            ImageIDList = idx_arr.tolist()

            # Super-Resolve the batch of validaiton data
            gen_valid = self.generator(X, training=False).numpy()

            reconstructed_catalog = fill_pixcatalog(gen_valid, reconstructed_catalog, ImageIDList, std_noise/1000)
            

            PSNR += compute_PSNR_batch(gen_valid, Y)/its
        # Convert the generated catalog to a Pandas DataFrame
        reconstructed_catalog = pd.DataFrame(reconstructed_catalog, columns=self.cat_cols).astype(self.dtypes)
        
        # Evaluation of source and flux metrics is not possible yet due to no detected sources in the super-resolved images
        if len(reconstructed_catalog["peak"]) == 0:
            return 0, 0, np.nan, np.nan, np.nan, PSNR


        # Compute completeness, reliability and the match catalog
        ## Warning: This function uses a parallel implementation. 3 CPU cores are required
        confusion_df, matches_catalog = get_matches_n_confusion(self.valid_catalog, reconstructed_catalog, confusion_df, matching_args, return_matches_df=True)

        # Calculate global completeness and reliability of test sample
        C = np.sum(confusion_df['TPc'])/(np.sum(confusion_df['TPc']) + np.sum(confusion_df['FNc']))
        R = np.sum(confusion_df['TPr'])/(np.sum(confusion_df['TPr']) + np.sum(confusion_df['FPr']))

        # When the number of detected generated sources exceed 1/6th of total number of true "known" sources, the C, R metrics start becoming useful
        thresh_bool = (np.sum(confusion_df['TPr']) + np.sum(confusion_df['FPr'])) >= 1/3*(np.sum(confusion_df['TPc']) + np.sum(confusion_df['FNc']))
        
        # We fit the global flux distribution and compute the MAPE of the global flux distribution
        func = lambda x, a, b: a*x + b
        self.printlog(f"{datetime.datetime.now()} - (thresh_bool_condl, thresh_bool_condr): {(np.sum(confusion_df['TPr']) + np.sum(confusion_df['FPr'])), 1/3*(np.sum(confusion_df['TPc']) + np.sum(confusion_df['FNc']))}")
        self.printlog(f"{datetime.datetime.now()} - (length cat): {reconstructed_catalog.shape}")
        self.printlog(f"{datetime.datetime.now()} - (target cat): {self.valid_catalog.shape}")
        if thresh_bool and C != 0 and reconstructed_catalog.shape[0]<=self.valid_catalog.shape[0]:
            a, b, Target_flux_bins, Reconstructed_flux_bins = fit_flux_distribution(matches_catalog, rnd_its, (self.valid_catalog, reconstructed_catalog), **coincidence_matching_args)
            d_a, d_b, = abs(a - 1.), abs(b - 0.)
            absolute_percentage_errors = np.abs((np.array(Target_flux_bins) - np.array(Reconstructed_flux_bins)) / np.array(Target_flux_bins))
            mape = np.mean(absolute_percentage_errors)
        else:
            d_a, d_b = np.nan, np.nan
            mape = np.nan
            C, R = np.nan, np.nan

        return C, R, d_a, d_b, mape, PSNR

    def Train(self):
        config_params = self.config['TRAINING PARAMETERS']

        self.BATCH_SIZE = int(config_params['batch_size'].strip())
        self.n_stopping_threshold = int(config_params['early_stopping'].strip())

        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # or tf.keras.optimizers.RMSprop(5e-5)
        
        # Graphed function performing a training step
        @tf.function()
        def train_step_GAN_G(X, Y):
            with tf.GradientTape() as gen_tape:
                gen_output = self.generator(X, training=True)
                gen_loss = non_adversarial_loss(gen_output, Y)
                fake_output = self.discriminator([gen_output, X], training=True)
                g_loss_fake = tf.cast(tf.math.reduce_mean(fake_output), tf.float32)
                G_loss_adv =  -g_loss_fake
                g_loss = 10*gen_loss + G_loss_adv
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            return
        
        @tf.function()
        def train_step_GAN_D(X, Y): #discriminator, discriminator_optimizer
            with tf.GradientTape() as disc_tape:
                gen_output = self.generator(X, training=True)
                
                real_output = self.discriminator([Y, X], training=True)
                fake_output = self.discriminator([gen_output, X], training=True)

                d_loss_real = tf.cast(tf.math.reduce_mean(real_output), tf.float32)#Wasserstein_loss(-tf.ones_like(real_output), real_output)
                d_loss_fake = tf.cast(tf.math.reduce_mean(fake_output), tf.float32)#Wasserstein_loss(tf.ones_like(fake_output), fake_output)

                gp = gradient_penalty(gen_output, Y)

                d_loss = -(d_loss_real - d_loss_fake) + 10*gp
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            return 
        
        @tf.function
        def gradient_penalty(gen_output, Y):
            epsilon = tf.random.normal([tf.shape(Y)[0], 1, 1, 1], 0.0, 1.0)
            x_hat = epsilon * Y + (1 - epsilon) *gen_output

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(x_hat)
                pred = self.discriminator([x_hat, X], training=True)
            grads = gp_tape.gradient(pred, [x_hat])[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gp = tf.reduce_mean((norm - 1.0)**2)
            return gp


        # Start Training
        self.n_epochs = int(config_params['number_of_epochs'].strip())
        self.n_epochs_arr = range(self.n_epochs)        
        
        # Training Hyperparameterization
        num_vis = 9
        n_stopping = 0
        rg = rd = 1.0
        λ = 10
        
        # Number of iterations per epoch
        its = self.train_arr_X.shape[0] // self.BATCH_SIZE
        EPOCH_THRESHOLD_MODELSAVING = 50  # Define the threshold for model saving

        # Initialize training history if necessary
        if not self.first_run:
            train_metrics_hist = np.load(os.path.join(self.model_path, 'TrainingLossHistory.npz'))
            valid_metrics_hist = np.load(os.path.join(self.model_path, 'ValidationLossHistory.npz'))
            valid_metrics = {}
            train_metrics = {}

            # Convert back to python dictionaries
            for key in valid_metrics_hist:
                valid_metrics[key] = valid_metrics_hist[key].tolist()

            for key in train_metrics_hist:
                train_metrics[key] = train_metrics_hist[key].tolist()

            # Remaining epoch numbers
            self.n_epochs_arr = np.arange(train_metrics["epochs"][-1] + 1, int(config_params['number_of_epochs'].strip()))

            # Best metric values
            best_metrics = np.load(os.path.join(self.model_path, 'Checkpoint_metrics.npz'))
            best_C = best_metrics["best_C"][0]
            best_d_a, best_d_b = best_metrics["best_d_a"][0], best_metrics["best_d_b"][0]
            best_mape = best_metrics["best_mape"][0]
            best_PSNR = best_metrics["best_PSNR"][0]
            n_stopping = best_metrics["n_stopping"][0]

        else:
            train_metrics = {"Non-Adversarial G loss":[], "Adversarial G loss":[], "D loss":[], "G score":[], "D score":[], "Gradient D":[], "rglambda":[], "rd":[], "G updates":[], "D updates":[], "epochs":[]}
            valid_metrics = {"PSNR":[], "d_a":[], "d_b":[], "mape":[], "Cglobal": [], "Rglobal": [], "epochs":[]}

        # Training Loop
        for epoch in tqdm(self.n_epochs_arr, desc="Training GAN..."):
            train_metrics["Non-Adversarial G loss"].append(0)
            train_metrics["Adversarial G loss"].append(0)
            train_metrics["D loss"].append(0)
            train_metrics["Gradient D"].append(0)
            train_metrics["G updates"].append(0)
            train_metrics["D updates"].append(0)
            train_metrics["G score"].append(0)
            train_metrics["D score"].append(0)
            train_metrics["epochs"].append(epoch)
            # Load all data batches, then apply training function
            # tf.keras.backend.clear_session()
            for i in range(its):
                # Adverserial training of discriminator/critic
                # Training of generator, if adversarial is enabled we also train with wasserstein loss
                X, Y= self.get_real_images()
                if rd > λ*rg:
                    train_step_GAN_D(X, Y)
                    train_metrics["D updates"][-1] += 1
                else:
                    train_step_GAN_G(X, Y)
                    train_metrics["G updates"][-1] += 1
                # Randomly sample a batch to evaluate current loss
                real_output = self.discriminator([Y, X], training=False)
                gen_output = self.generator(X, False)
                fake_output = self.discriminator([gen_output, X], training=False)

                # Calculate the current losses
                D_loss_real = tf.math.reduce_mean(real_output)
                D_loss_fake = tf.math.reduce_mean(fake_output)
                G_loss_adv = -D_loss_fake
                gen_loss = non_adversarial_loss(gen_output, Y)
                Lg_current = 10*gen_loss + G_loss_adv
                gp = gradient_penalty(gen_output, Y).numpy()
                Ld_current = -(D_loss_real - D_loss_fake) + 10*gp

                if epoch == self.n_epochs_arr[0] and i == 0:
                    Lg_prev = Lg_current
                    Ld_prev = Ld_current
                
                rg, rd = float(tf.abs((Lg_current - Lg_prev)/Lg_prev).numpy()), float(tf.abs((Ld_current - Ld_prev)/Ld_prev).numpy())
                Lg_prev, Ld_prev = Lg_current, Ld_current

                train_metrics["Non-Adversarial G loss"][-1] += float(gen_loss.numpy()/its)
                train_metrics["Adversarial G loss"][-1] += float(G_loss_adv/its)
                train_metrics["D loss"][-1] += float(Ld_current/its)
                train_metrics["G score"][-1] += float(D_loss_fake/its)
                train_metrics["D score"][-1] += float(D_loss_real/its) 
                train_metrics["Gradient D"][-1] += float((np.sqrt(gp)+1)/its)
               
            # Write training progress to log
            if self.verbose == True:
                self.printlog(f"{datetime.datetime.now()} - Epoch: {train_metrics['epochs'][-1]}")
                self.printlog(f"{datetime.datetime.now()} - Non-Adversarial loss: {train_metrics['Non-Adversarial G loss'][-1]:.3f}")
                self.printlog(f"{datetime.datetime.now()} - Adversarial Score (G, D, Wdistance): ({train_metrics['G score'][-1]:.2f}, {train_metrics['D score'][-1]:.2f}, {(train_metrics['D score'][-1] - train_metrics['G score'][-1]):.2f})")
                self.printlog(f"{datetime.datetime.now()} - Last iteration (rg, rd): {(rg, rd)}")
                self.printlog(f"{datetime.datetime.now()} - Training Updates (G, D): {(train_metrics['G updates'][-1], train_metrics['D updates'][-1])}")
                self.printlog(f"{datetime.datetime.now()} - Discriminator Gradient: {train_metrics['Gradient D'][-1]:.2f}")
                self.printlog(f"{datetime.datetime.now()} - n_stopping: {n_stopping}")

            # We only start evaluating the validation set after a few iterations and only every 5 epochs
            if epoch >= EPOCH_THRESHOLD_MODELSAVING and epoch % 10 == 0:
                # Evaluate the validation set
                C, R, d_a, d_b, mape, PSNR = self.evaluate_metrics()

                valid_metrics["PSNR"].append(PSNR)
                valid_metrics["d_a"].append(d_a)
                valid_metrics["d_b"].append(d_b)
                valid_metrics["mape"].append(mape)
                valid_metrics["Cglobal"].append(C)
                valid_metrics["Rglobal"].append(R)
                valid_metrics["epochs"].append(epoch)
                # After EPOCH_THRESHOLD_MODELSAVING we will start saving the best model
                if epoch == EPOCH_THRESHOLD_MODELSAVING:
                    best_PSNR = PSNR
                    n_stopping = 0 # Reset stopping criteria and save models
                    self.generator.save(os.path.join(self.model_path, 'BestPSNR_Model')) # Save the best model
                    if self.verbose == True:
                        self.printlog(f"{datetime.datetime.now()} - Average PSNR Improved! Average PSNR: {PSNR}")
                        self.printlog(f"{datetime.datetime.now()} - Best PSNR Model saved!")

                    # Flux reproduction model    
                    if not np.isnan(d_a):
                        best_C = C
                        best_d_a, best_d_b = d_a, np.round(d_b)
                        best_mape = mape      
                        self.generator.save(os.path.join(self.model_path, 'BestFluxReproduction_Model'))
                        if self.verbose == True:
                            self.printlog(f"{datetime.datetime.now()} - Flux Reproduction Improved! d_a: {d_a}, d_b: {d_b}, mape: {mape}")
                            self.printlog(f"{datetime.datetime.now()} - Confusion Score Improved! C: {C}, R: {R}")
                            self.printlog(f"{datetime.datetime.now()} - Best FluxReproduction Model saved!")

                    else:
                        #  Insanely high values, because the results are not reliable yet
                        best_C = 0.
                        best_d_a, best_d_b = 1000, 1000
                        best_mape = 1.

                elif epoch > EPOCH_THRESHOLD_MODELSAVING:
                    if not np.isnan(d_a):
                        # Check which params improved, and save model accordngly
                        if valid_metrics["d_a"][-1] <= best_d_a and np.round(valid_metrics["d_b"][-1]) <= best_d_b and mape <= best_mape:
                            n_stopping = 0 # Reset stopping criteria
                            best_d_a, best_d_b, best_mape = valid_metrics["d_a"][-1], np.round(valid_metrics["d_b"][-1]), mape
                            self.generator.save(os.path.join(self.model_path, 'BestFluxReproduction_Model'))
                            if self.verbose == True:
                                self.printlog(f"{datetime.datetime.now()} - Flux Reproduction Improved! d_a: {best_d_a}, d_b: {best_d_b}, mape: {best_mape}")
                                self.printlog(f"{datetime.datetime.now()} - Best FluxReproduction Model saved!")

                        if valid_metrics["Cglobal"][-1] >= best_C:
                            best_C = C
                            if self.verbose == True:
                                self.printlog(f"{datetime.datetime.now()} - Confusion Score Improved! C: {valid_metrics['Cglobal'][-1]}, R: {valid_metrics['Rglobal'][-1]}")


                    if valid_metrics["PSNR"][-1] > best_PSNR:
                        best_PSNR = valid_metrics["PSNR"][-1]
                        n_stopping = 0 # Reset stopping criteria
                        self.generator.save(os.path.join(self.model_path, 'BestPSNR_Model'))
                        if self.verbose == True:
                            self.printlog(f"{datetime.datetime.now()} - Average PSNR Improved! Average PSNR: {best_PSNR}")
                            self.printlog(f"{datetime.datetime.now()} - Best PSNR Model saved!")

                # Save the checkpoint model, which can be used to continue training later on again.
                self.generator.save(os.path.join(self.model_path, 'checkpoint_GModel'))
                self.discriminator.save(os.path.join(self.model_path, 'checkpoint_DModel'))
                best_metrics = {"best_PSNR":[best_PSNR], "best_d_a":[best_d_a], "best_d_b":[best_d_b], "best_mape":[best_mape], "best_C": [best_C], "n_stopping":[n_stopping]}
                np.savez(os.path.join(self.model_path, 'Checkpoint_metrics.npz'), **best_metrics)
                np.savez(os.path.join(self.model_path, 'ValidationLossHistory.npz'), **valid_metrics)
                np.savez(os.path.join(self.model_path, 'TrainingLossHistory.npz'), **train_metrics)

                if self.verbose == True:
                    self.printlog(f"{datetime.datetime.now()} - Checkpoint model saved!")
                    self.printlog(f"{datetime.datetime.now()} - Best metrics saved!")

            n_stopping += 1

            if n_stopping >= self.n_stopping_threshold:
                break

            # Memory CleanUp
            if (epoch) % 20 == 0:
                gc.collect()

            # Generate snapshot of the same selected images of the validation set every epoch
            if epoch%10 == 0:
                TrainingSnapShot(self.generator, epoch, self.valid_arr_X[:9], self.tdir_out_progress)
                TrainingSnapShotFeatureSample(self.generator, self.valid_arr_X[:1], self.valid_arr_Y[:1], epoch, labels=["24", "250", "350", "500", "SR500", "Reconstructed"], save_path=self.tdir_out_progress)

        # Save the training and validation history
        np.savez(os.path.join(self.model_path, 'ValidationLossHistory.npz'), **valid_metrics)
        np.savez(os.path.join(self.model_path, 'TrainingLossHistory.npz'), **train_metrics)
        self.printlog(f"{datetime.datetime.now()} - Stored Training/Validation Loss History!")

    def TrainingAnalysis(self):
        self.printlog(f"{datetime.datetime.now()} - Analysing Training Run!")
        # Make plots that describe the model's performance during training
        losshist = np.load(os.path.join(self.model_path, 'TrainingLossHistory.npz'))
        # Plot the training history of various variables that have been tracked
        plot_line_chart(losshist["epochs"], [losshist["Non-Adversarial G loss"]], "Epoch Number", r"$\mathcal{L}_G^{non-adv}$", labels=[""], save_path=os.path.join(self.tdir_out_analysis, 'non_adv_G_loss.pdf'), log_scale=True)
        plot_line_chart(losshist["epochs"], [losshist["G score"], losshist["D score"], losshist["Adversarial G loss"]], "Epoch Number", r"$\mathcal{L}^{adv}$", labels=[r"Generator Score", r"Discriminator Score", r"Wasserstein Distance: $\mathcal{L}^{adv}_G$"], save_path=os.path.join(self.tdir_out_analysis, 'Adversarial_loss.pdf'), log_scale=False)
        plot_line_chart(losshist["epochs"], [losshist["G updates"], losshist["D updates"]], "Epoch Number", "$N_{updates}$ per epoch", labels=[r"Generator", r"Discriminator"], save_path=os.path.join(self.tdir_out_analysis, 'Adversarial_updates.pdf'), log_scale=False)
        plot_line_chart(losshist["epochs"], [losshist["Gradient D"]], "Epoch Number", r"$\Vert \nabla D(\hat{x}) \Vert$", labels=[""], save_path=os.path.join(self.tdir_out_analysis, 'discriminator_gradient.pdf'), log_scale=False)
        
        del losshist;

        valhist = np.load(os.path.join(self.model_path, 'ValidationLossHistory.npz'))
        # Plot the validation history of various variables that have been tracked
        plot_line_chart(valhist["epochs"], [valhist["PSNR"]], "Epoch Number", r"$\left<PSNR\right> [dB]$", labels=[""], save_path=os.path.join(self.tdir_out_analysis, 'PSNR.pdf'), log_scale=False)
        plot_line_chart(valhist["epochs"], [valhist["d_a"]], "Epoch Number", r"$|1. - a|$", labels=[""], save_path=os.path.join(self.tdir_out_analysis, 'd_a.pdf'), log_scale=False)
        plot_line_chart(valhist["epochs"], [valhist["d_b"]], "Epoch Number", r"$|1. - b|$", labels=[""], save_path=os.path.join(self.tdir_out_analysis, 'd_b.pdf'), log_scale=False)
        plot_line_chart(valhist["epochs"], [valhist["mape"]], "Epoch Number", r"$\frac{|S_{500, target} - S_{500, SR}|}{S_{500, SR}}$", labels=[""], save_path=os.path.join(self.tdir_out_analysis, 'mape.pdf'), log_scale=True)        
        plot_line_chart(valhist["epochs"], [valhist["Cglobal"], valhist["Rglobal"]], "Epoch Number", r"Percentage [%]", labels=["Completeness", "Reliability"], save_path=os.path.join(self.tdir_out_analysis, 'CR.pdf'), log_scale=True)

        anim_file = os.path.join(self.tdir_out_analysis, 'dcgan.gif')
        anim_file2 = os.path.join(self.tdir_out_analysis, 'WGANGP_feature_animation.gif')
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.join(self.tdir_out_progress, 'image_at_epoch*.png'))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)


        with imageio.get_writer(anim_file2, mode='I') as writer:
            filenames = glob.glob(os.path.join(self.tdir_out_progress, 'feature_sample_at_epoch*.png'))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        self.printlog(f"{datetime.datetime.now()} - Analysis Complete!")