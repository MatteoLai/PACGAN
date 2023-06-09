"""
Non-Profit Open Software License 3.0 (NPOSL-3.0)
Copyright (c) 2023 Matteo Lai

@authors: 
         Matteo Lai, M.Sc, Ph.D. student
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: matteo.lai3@unibo.it

         Chiara Marzi, Ph.D., Junior Assistant Professor
         Department of Statistics, Computer Science, Applications "G. Parenti" (DiSIA)
         University of Firenze, Firenze, Italy
         Email address: chiara.marzi@unifi.it
         
         Luca Citi, Ph.D., Professor
         School of Computer Science and Electronic Engineering
         University of Essex, Colchester, UK
         Email address: lciti@essex.ac.uk
         
         Stefano Diciotti, Ph.D, Associate Professor
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: stefano.diciotti@unibo.it
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import os
import math
import time
from datetime import datetime
from math import log2

from generate_images import Generate
import argparse
import json
from utils import (create_folder,
    load_model, restore_model,
    seed_everything,
    get_loader, catch)
from train import train_fn
from model import Discriminator, Generator
from assessment import Quantitative_metrics
seed=8
seed_everything(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--jsonfile',
        dest='JSONfilename',
        help='JSON configuration file',
        metavar='FILE',
        default='PACGAN/Training/config.json')
args = parser.parse_args()

def main():

    # READ CONFIG FILE ------------------------------------------------------------------------------------

    with open(args.JSONfilename) as f:
        config_data = json.load(f)

    # Set the path to data directories
    data_train_dir = config_data["DATA_TRAIN_DIRECTORY"]
    labels_train_dir = config_data["LABEL_TRAIN_DIRECTORY"]
    data_test_dir = config_data["DATA_TEST_DIRECTORY"]
    labels_test_dir = config_data["LABEL_TEST_DIRECTORY"]

    # Set the path to saving directories
    images_directory = config_data["IMAGE_DIRECTORY"]
    model_directory = config_data["MODEL_DIRECTORY"]

    # Set if you want to work with grayscale or RGB images
    img_channels = config_data["IMAGE_CHANNELS"]

    # Set the number of classes of the labeled images
    class_size = config_data["CLASS_SIZE"]

    # Set the device to use for the training
    device = eval(config_data["DEVICE"])
    # If cuda is available, set which GPUs use:
    gpus = eval(config_data["GPUS_N"])

    # Set the architechture of the progressive growing network
    in_channels = config_data["IN_CHANNELS"]
    factors = eval(config_data["FACTORS"])

    # Set training hyperparameters
    lr = config_data["LEARNING_RATE"]
    bs = eval(config_data["BATCH_SIZES"])
    embedding_dim = config_data["EMBEDDING_DIM"]
    z_dim = config_data["Z_DIM"]
    critic_iterations = eval(config_data["CRITIC_ITERATIONS"])
    L_GP = config_data["LAMBDA_GP"]
    L_cls1 = config_data["LAMBDA_CLS_1"]
    L_cls2 = eval(config_data["LAMBDA_CLS_2"])

    # Set the number of epoch at each resolution level
    prog_epochs = eval(config_data["PROGRESSIVE_EPOCHS"])

    # Set how often do you want to save the images during the training
    disp_every_n_epochs = config_data["DISP_EVERY_N_EPOCHS"]

    # Set if you want to measure the performance on validation set and, in case, from which epoch you want to start ckecking the performance
    validate = eval(config_data["VALIDATE"])
    perc = config_data["START_CHECK_AT"]
    epoch_checking = [i*perc for i in prog_epochs]

    # Set if you want to measure the performances on test set and, in case, if you want to exploit the best_model found during validation or if you want to train for prog_epochs
    testing = eval(config_data["TESTING"])
    use_best_model = eval(config_data["TEST_USING_BEST_MODEL"])

    # Set if you want to generate images, and the number of images for each class
    generate = eval(config_data["GENERATE"])
    gen_n_images = eval(config_data["N_IMAGES_GENERATED"])

    # Set if you want to start the training with a pre-trained model
    load_mod = eval(config_data["LOAD_MODEL"])
    start_train_at_img_size = config_data["START_TRAIN_AT_IMG_SIZE"]

    # If you are testing, save the results in a different folder
    if testing and not validate:
        images_directory = images_directory+'_test'; create_folder(images_directory)
        model_directory = model_directory+'_test'; create_folder(model_directory)

    # ------------------------------------------------------------------------------------------------
    
    catch(data_train_dir, labels_train_dir, start_train_at_img_size, bs, critic_iterations, prog_epochs, class_size)

    start_datetime = datetime.now()
    print('Current DateTime:', start_datetime, flush=True)
    time_start = time.time()

    # Initialize models
    gen = Generator(z_dim, embedding_dim, class_size, img_channels, in_channels, factors)
    gen.to(device)
    critic = Discriminator(img_channels, in_channels, factors)
    critic.to(device)

    if len(gpus)>1:
        gen = nn.DataParallel(gen, device_ids = gpus)
        critic = nn.DataParallel(critic, device_ids = gpus)

    # initialize optimizers and scalers for training
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # Divide the dataset in training e validation
    if validate:
        # To keep 20% of data for the validation, perform the 5-fold CV and keep the first split
        labels_df = pd.read_csv(labels_train_dir)
        X = labels_df['ID']
        y = labels_df['Label']
        groups = labels_df['Subject_ID']
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        print('Extracting validation set...', flush=True)
        for train_idxs, val_idxs in cv.split(X, y, groups):
            pass

    # Start at step that corresponds to img size that you set in config.json
    start_res = start_train_at_img_size if load_mod else 4
    step = int(log2(start_res / 4)) #step=0 (4x4), step=1 (8x8), ...
    for step_idx, num_epochs in enumerate(prog_epochs[step:]):
        img_size = 4 * 2 ** step

        # Load the data
        if validate:
            # Training and validation set
            train_loader, train_dataset = get_loader(step, data_train_dir, labels_train_dir, bs[step], len(gpus), train_idxs)
            val_loader, val_dataset = get_loader(step, data_train_dir, labels_train_dir, bs[step], len(gpus), val_idxs)
            test_loader, test_dataset = None, None
        else:
            # Training+validation and test set
            train_loader, train_dataset = get_loader(step, data_train_dir, labels_train_dir, bs[step], len(gpus))
            val_loader, val_dataset = None, None
            if testing:
                test_loader, test_dataset = get_loader(step, data_test_dir, labels_test_dir, bs[step], len(gpus))
            else:
                test_loader, test_dataset = None, None

        # Load models
        if step_idx == 0:
            if load_mod and start_train_at_img_size>4:
                start_step = int(log2(start_train_at_img_size / 4))
                img_size_prec = 4 * 2 ** (start_step - 1)
                print('\nStarting training with image size {}'.format(img_size))
                load_model(gen, opt_gen, scaler_gen, img_size_prec, model_directory, 'generator_best.pt', device, len(gpus))
                load_model(critic, opt_critic, scaler_critic, img_size_prec, model_directory, 'discriminator_best.pt', device, len(gpus))
        else:   
            # Beside the first step (4x4), load the best models of the prevoius resolution
            print('\nUploading best models of previous step...')
            img_size_prec = 4 * 2 ** (step -1)
            restore_model(gen, opt_gen, scaler_gen, lr, img_size_prec, model_directory, 'generator_best.pt', device, gpus)
            restore_model(critic, opt_critic, scaler_critic, lr, img_size_prec, model_directory, 'discriminator_best.pt', device, gpus)
        print('\nCurrent image size: {}'.format(img_size))

        # Initialize alpha, starting with a very low value
        alpha = 1e-5

        ncritic = critic_iterations[step]

        train_fn(
            critic, gen,
            opt_critic, opt_gen,  
            scaler_critic, scaler_gen,
            train_loader, train_dataset,
            validate, epoch_checking,
            val_loader, val_dataset,
            testing, use_best_model,
            test_loader, test_dataset,
            step, 
            alpha, 
            num_epochs,
            img_size, 
            class_size,
            ncritic,
            z_dim,
            L_GP, L_cls1, L_cls2,
            model_directory, images_directory,
            device, gpus,
            disp_every_n_epochs
)

        # Progress to the next img size
        step += 1  
    
    print('\nTraining finished.', flush=True)

    if generate:
        # Generate n_images and save them on a nifti file
        img_size = 4 * 2 ** (step-1)
        model_path =  os.path.join(model_directory, 'PACGAN_{}x{}'.format(img_size,img_size))
        image_path = os.path.join(images_directory,'PACGAN_{}x{}'.format(img_size,img_size))
        best_model_path =  os.path.join(model_path, 'generator_best.pt')
        best_image_path = os.path.join(image_path,'Generated{}x{}.nii.gz'.format(img_size,img_size))
        Generate(z_dim, embedding_dim, class_size, img_channels, in_channels, factors, img_size, best_model_path, best_image_path, device, gpus).Gen(gen_n_images)

        # Compute quantitative metrics (FID, KID, SSIM)
        real_path = os.path.join(data_test_dir,'Real256x256.nii.gz')
        real_y_path = labels_test_dir
        fake_y_path = os.path.join(image_path,'labels_syn.csv')
        
        # For the best model
        fake_path_BestModel = os.path.join(image_path,'Generated{}x{}.nii.gz'.format(img_size,img_size))
        txt_name = "metrics.txt" if validate else "metrics_test.txt"
        output_txt = os.path.join(images_directory, txt_name)
        Quantitative_metrics(real_path, fake_path_BestModel, real_y_path, fake_y_path, img_channels, class_size, output_txt, device, 50).compute()

    print('\nStart DateTime:  ', start_datetime, flush=True)
    print('Current DateTime:', datetime.now(), flush=True)
    time_end = time.time() - time_start
    print('\n Training time [s]: {}\n'.format(time_end), flush=True)
    m, h = math.modf(time_end/3600)
    print(' --> {} hours and {} minutes.'.format(int(h), int(m*60)), flush=True)

    # ---------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
