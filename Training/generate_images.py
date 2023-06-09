"""
Non-Profit Open Software License 3.0 (NPOSL-3.0)
Copyright (c) 2023 Matteo Lai

@authors: Matteo Lai, Ph.D. student
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: matteo.lai3@unibo.it

         AI for Medicine Research Group
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
"""

import nibabel as nib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from math import log2

from model import Generator

from utils import seed_everything
seed_everything(0)

class Generate():
    """
    Generates "n_images" images and saves them on a nifti file. 
    The first half has label 0, the other half has label 1.
    """

    def __init__(
        self,
        z_dim: int,                     # Dimension of the latent space of the generator
        embedding_dim: int,             # Dimension of the embedding space of the generator
        class_size: int,                # Number of classes
        img_channels: int,              # Number of channels of the images (1 for grayscale, 3 for RGB)
        in_channels: int,               # Number of channels of the input of the generator
        factors: list,                  # List of factors used for scaling in the progressive growing structure (e.g. [1, 1, 1/2, 1/4, 1/8, 1/16, 1/32])
        img_size: int,                  # Size of the images to generate
        model_path: str,                # Path of the pre-trained generator
        image_path: str,                # Path where to save the generated images
        device: str = 'cpu',            # 'cpu' or 'cuda'
        gpus: list = [0]                # List of the GPUs used to train the generator
    ):

        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        self.class_size = class_size
        self.d = img_size
        self.step = int(log2(img_size/ 4))
        self.image_path = image_path
        self.device = device
        self.gpus = gpus

        # Define the path of the output
        head_path, _ = os.path.split(self.image_path)
        self.path_labels = os.path.join(head_path, 'labels_syn.csv')

        # Define Generator model
        self.G = Generator(z_dim, embedding_dim, class_size, img_channels, in_channels, factors)
        self.G.to(self.device)

        # Load the state_dict of the pre-trained generator
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # If the model was trained on multi-GPUs, remove the "module." from the key names 
        new_state_dict = state_dict.copy()
        for key, param in state_dict.items():
            if key.startswith('module.'):        
                new_state_dict[key[7:]] = param          
                new_state_dict.pop(key)

        # Check that the architecture of the model defined correspond with the loaded one:
        if new_state_dict['label_embedding.weight'].shape[0] != self.class_size:
            print('\nError: trying to load a model with CLASS_SIZE = {} on an architechture with CLASS_SIZE = {}'.format(new_state_dict['label_embedding.weight'].shape[0], self.class_size), flush=True)
            print('Please check the settings in config.py\n', flush=True)
            sys.exit()
        if new_state_dict['label_embedding.weight'].shape[1] != self.embedding_dim:
            print('\nError: trying to load a model with EMBEDDING_DIM = {} on an architechture with EMBEDDING_DIM = {}'.format(new_state_dict['label_embedding.weight'].shape[1], self.embedding_dim), flush=True)
            print('Please check the settings in config.py\n', flush=True)
            sys.exit()            
        if (new_state_dict['initial_block.1.weight'].shape[0] - self.embedding_dim) != self.z_dim:
            print('\nError: trying to load a model with Z_DIM = {} on an architechture with Z_DIM = {}'.format((new_state_dict['initial_block.1.weight'].shape[0]-self.embedding_dim), self.z_dim), flush=True)
            print('Please check the settings in config.py\n', flush=True)
            sys.exit()
        if  not(img_size in [4, 8, 16, 32, 64, 128, 256]):
            print('\nError: image_size can only be one of the following: 4, 8, 16, 32, 64, 128, 256', flush=True)
            sys.exit()
            
        # Load the state_dict of the generator
        self.G.load_state_dict(new_state_dict)

        if self.device=='cuda' and len(self.gpus)>1:
            self.G = nn.DataParallel(self.G, device_ids = self.gpus)
        
        # Set the generator in inference mode
        self.G.eval()

    def Gen(self, 
            n_images_xCLASS: list     #number of images to generate for each class
            ):
        """
            The list n_images_xCLASS can contain 
            > one element (defining the total number of images to generate).
                In this case, the first half will have label 0, the other half will have label 1.
            > a number of elements equal to the number of classes (defining the number of images to generate for each class) 

        """
        if len(n_images_xCLASS) == 1:
            # Define the labels to generate an equal number of images for each class
            n_images = int(n_images_xCLASS[0])
            n_images_xCLASS = int(n_images/self.class_size)
            labels = torch.tensor([],dtype=int).to(self.device)
            for i in range(self.class_size):
                labels = torch.cat((labels, torch.tensor([i]*n_images_xCLASS,dtype=int).to(self.device))).to(self.device)
            labels = torch.cat((labels, torch.tensor([i]*(n_images-len(labels)),dtype=int).to(self.device))).to(self.device)

        elif len(n_images_xCLASS) == self.class_size:
            # Define the labels to generate, for each class, a number of images as set by n_images_xCLASS
            n_images = sum(n_images_xCLASS)
            labels = torch.tensor([],dtype=int).to(self.device)
            for i in range(self.class_size):
                labels = torch.cat((labels, torch.tensor([i]*n_images_xCLASS[i],dtype=int).to(self.device))).to(self.device)

        else:
            print('Error: define the number of images to generate for each of the {} classes'.format(self.class_size))
            sys.exit()

        print('\nSaving generated images in a nifti file...', flush=True)

        # Generate new images:
        z = torch.randn(n_images, self.z_dim,1,1).to(self.device)
        alpha = 1 # Consider the generator at the end of the progressive growing process
        with torch.no_grad():
            Images = self.G(z, labels, alpha, self.step)
    
        # Save the nifti file
        Images_array = Images.cpu().detach().numpy()            # [n_images, 1, d, d]
        Images_array = np.swapaxes(Images_array, 0,3)
        Images_array = np.swapaxes(Images_array, 1,2)
        nifti_file = nib.Nifti1Image(Images_array, np.eye(4))   # [d, d, 1, n_images]
        nib.save(nifti_file, self.image_path)
        print('\n=> Saved the synthetic images in {}'.format(self.image_path), flush=True)

        # Define the output csv
        col = ['ID', 'Label']
        labels_df = pd.DataFrame(columns=col)
        labels_df['Label'] = labels.cpu().detach().numpy()
        labels_df['ID'] = labels_df.index
        labels_df.to_csv(self.path_labels, index=False)
        print('=> Saved the labels of the synthetic images in {}'.format(self.path_labels), flush=True)

        return nifti_file
