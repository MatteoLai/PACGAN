import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import sys
import json
import os
from math import log2

from model import Generator
from utils import seed_everything
seed_everything(0)


parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=256,
                    help="Size of the image you want to generate.")
parser.add_argument("--n_images_xCLASS", nargs="*", type=int, 
                    help="Number of images to generate for each class")
parser.add_argument("--model_path", type=str,
                    help="Path to the generator model.")
parser.add_argument("--images_path", type=str,
                    help="Path to the folder where to save generated images.")
parser.add_argument("--device", type=str, default='cpu',
                    help="Device to use to do inference. By default, device='cpu'")
parser.add_argument("--gpus", nargs="*", type=int, default=[0],
                    help="GPU(s) to use to do inference. By default, gpus=[0]")
parser.add_argument('-j', '--jsonfile',
                dest='JSONfilename',
                help='JSON configuration file',
                metavar='FILE',
                default='PACGAN/config.json')

args = parser.parse_args()

img_size = args.img_size
model_path = args.model_path
images_path = args.images_path
n_images_xCLASS = args.n_images_xCLASS
device = args.device
gpus = args.gpus

class Generate():
    def __init__(
        self,
        img_size: int,              # Size of the images to generate
        model_path: str,            # Path of the pre-trained generator
        image_path: str,            # Path where to save the generated images
        device: str = 'cpu',        # Device to use for inference ('cpu' or 'cuda')
        gpus: list = [0]            # List of the GPUs used to train the generator
    ):
        with open(args.JSONfilename) as f:
            config_data = json.load(f)

        self.z_dim = config_data["Z_DIM"]
        self.embedding_dim = config_data["EMBEDDING_DIM"]
        self.class_size = config_data["CLASS_SIZE"]
        self.d = img_size
        self.step = int(log2(img_size/ 4))
        self.image_path = image_path
        self.device = device
        self.gpus = gpus
        img_channels = config_data["IMAGE_CHANNELS"]
        in_channels = config_data["IN_CHANNELS"]
        factors = eval(config_data["FACTORS"])

        # Define the path of the output
        head_path, _ = os.path.split(self.image_path)
        self.path_labels = os.path.join(head_path, 'labels_syn.csv')

        # Define Generator model
        self.G = Generator(self.z_dim, self.embedding_dim, self.class_size, img_channels, in_channels, factors)
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
            print('\nError: trying to load a model with CLASS_SIZE = {} on an architecture with CLASS_SIZE = {}'.format(new_state_dict['label_embedding.weight'].shape[0], self.class_size), flush=True)
            print('Please check the settings in config.json\n', flush=True)
            sys.exit()
        if new_state_dict['label_embedding.weight'].shape[1] != self.embedding_dim:
            print('\nError: trying to load a model with EMBEDDING_DIM = {} on an architecture with EMBEDDING_DIM = {}'.format(new_state_dict['label_embedding.weight'].shape[1], self.embedding_dim), flush=True)
            print('Please check the settings in config.json\n', flush=True)
            sys.exit()            
        if (new_state_dict['initial_block.1.weight'].shape[0] - self.embedding_dim) != self.z_dim:
            print('\nError: trying to load a model with Z_DIM = {} on an architecture with Z_DIM = {}'.format((new_state_dict['initial_block.1.weight'].shape[0]-self.embedding_dim), self.z_dim), flush=True)
            print('Please check the settings in config.json\n', flush=True)
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

# Generate new synthetic images
Generate(img_size, model_path, images_path, device, gpus).Gen(n_images_xCLASS)


