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
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import os
import random
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class LoadDataset(Dataset):
    """
    Load the dataset.
    It takes as input images in range [-1,1]
    """

    def __init__(
        self,
        batch_size: int,                # Dimension of the batch size
        step: int,                      # Step of the progressive growing structure
        root: str,                      # Path to the images
        label_root: str,                # Path to the labels
        n_devices: int,                 # Number of GPUs used for training
        sel_ids: list = []              # List of the indexes of the images to consider (useful to divide training and validation sets)
    ):
        self.root = root
        image_size = 4 * 2 ** step

        # Load labels from file csv
        labels_data = pd.read_csv(label_root, delimiter=',')
        if sel_ids==[]:
            # If sel_ids (selected indexes) are not inserted as input, consider all the indexes
            sel_ids = labels_data['ID'].values.astype(np.int)
        self.labels = labels_data['Label'].values.astype(np.int)[sel_ids]

        root = os.path.join(root,'Real{}x{}.nii.gz'.format(image_size,image_size))

        # Load nifti data
        self.img = nib.load(root)
        self.img = np.asanyarray(self.img.dataobj) # numpy array of shape (W,H,C,N)
        self.img = np.float32(self.img)
        self.img = self.img[:,:,:,sel_ids]

        # Pytorch dataloader expect image of size N,C,H,W (batch_size, channel, height, width):
        self.img=np.swapaxes(self.img, 0,3)
        self.img=np.swapaxes(self.img, 1,2)   # after swapping axes, array shape (N,C,H,W)

        if n_devices>1:
            # Check to have a number of images multiple of the batch size (needed for parallelization):
            mod = len(sel_ids)%batch_size
            if mod!=0:
                self.img = self.img[:-mod,:,:,:]
                self.labels = self.labels[:-mod]

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __getitem__(self, index):
        return self.img[index,:,:,:], self.labels[index]

    def __len__(self):
        return self.img.shape[0] # image shape: [batch_size x 1 x image_size x image_size]
