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

import nibabel as nib
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
from math import log2
from sklearn.metrics import (roc_auc_score, confusion_matrix)

from model import Discriminator
import argparse
from utils import (seed_everything, show_confusion_matrix)
seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str,
                    help="Path to the nifti file.")
parser.add_argument("--model_path", type=str,
                    help="Path to the discriminator model.")
parser.add_argument("--device", type=str, default='cpu',
                    help="Device to use to do inference. By default, device='cpu'")
parser.add_argument("--gpus", nargs="*", type=int, default=[0],
                    help="GPU(s) to use to do inference. By default, gpus=[0]")
parser.add_argument('-j', '--jsonfile',
                dest='JSONfilename',
                help='JSON configuration file',
                metavar='FILE',
                default='Training/config.json')
args = parser.parse_args()

class Discriminate():
    """
    Classify the input images and save the predictions in a 'predictions.csv'
    """

    def __init__(
        self,
        img_path: str,              # Path of the input image
        model_path: str,            # Path of the pre-trained discriminator
        device: str = 'cpu',        # 'cpu' or 'cuda'
        gpus: list = [0]            # List of the GPUs used to train the discriminator
    ):
        with open(args.JSONfilename) as f:
            config_data = json.load(f)
        img_channels = config_data["IMAGE_CHANNELS"]
        in_channels = config_data["IN_CHANNELS"]
        factors = eval(config_data["FACTORS"])

        # Load the images
        self.img_path = img_path
        self.img = nib.load(self.img_path)
        self.img = np.asanyarray(self.img.dataobj) # numpy array of shape (W,H,C,N)
        self.img = np.float32(self.img)
        self.device = device
        self.gpus = gpus

        # Pytorch expect image of size N,C,H,W (batch_size, channel, height, width):
        self.img = np.swapaxes(self.img, 0,3)
        self.img = np.swapaxes(self.img, 1,2)   # after swapping axes, array shape (N,C,H,W)
        self.img = torch.tensor(self.img).to(self.device)

        # Set parameters
        self.img_size = self.img.shape[3]
        self.step = int(log2(self.img_size/ 4))

        # Define the path of the output
        head_path, _ = os.path.split(img_path)
        self.path_predictions = os.path.join(head_path,'predictions.csv')

        # Define Discriminator model
        self.D = Discriminator(img_channels, in_channels, factors)
        self.D.to(self.device)

        # Load the state_dict of the pre-trained discriminator
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        # If the model was trained on multi-GPUs, remove the "module." from the key names 
        new_state_dict = state_dict.copy()
        for key, param in state_dict.items():
            if key.startswith('module.'):        
                new_state_dict[key[7:]] = param          
                new_state_dict.pop(key)

        # Load the state_dict of the discriminator
        self.D.load_state_dict(new_state_dict)

        if self.device=='cuda' and len(self.gpus)>1:
            self.D = nn.DataParallel(self.D, device_ids = self.gpus)
        
        # Set the discriminator in inference mode
        self.D.eval()

    def Dis(self):
        alpha=1
        with torch.no_grad():
            _, cls = self.D(self.img, alpha, self.step)

        # Pass the output of the discriminator to the SoftMax activation function to obtain probability values
        SoftMax = nn.Softmax(dim=1)
        cls = SoftMax(cls)

        # Define the output csv
        col = ['Class {}'.format(i) for i in range(cls.shape[1])]
        predictions = pd.DataFrame(cls.cpu().detach().numpy(), columns=col)
        predictions.insert(0,'Predictions for {}'.format(self.img_path),cls.cpu().detach().numpy().argmax(axis=1))
        predictions.to_csv(self.path_predictions)
        print('\n=> Saved the predictions in {}'.format(self.path_predictions), flush=True)

# Classify new instances:
img_path = args.img_path
model_path = args.model_path
device = args.device
gpus = args.gpus

Discriminate(img_path, model_path, device, gpus).Dis()
