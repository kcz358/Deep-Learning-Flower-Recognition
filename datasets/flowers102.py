from glob import glob
import os

import numpy as np
from scipy.io import loadmat
import torch
from torchvision import transforms
from PIL import Image

from .base_dataset import BaseDataset

class Flowers102(BaseDataset):
    def __init__(self, 
                 dataset_name : str, 
                 dataset_path : str, 
                 split : str = 'train', 
                 transform = None,
                 n_shot : int = -1) -> None:
        super().__init__(dataset_name)
        self.dataset_path = dataset_path
        print(f"Debug: dataset_path = {dataset_path}")
        assert os.path.exists(dataset_path + "/jpg"), "No jpg (image) files in the folder"
        assert os.path.exists(dataset_path + "/imagelabels.mat"), "No label files in the folder"
        assert os.path.exists(dataset_path + "/setid.mat"), "No dataset split files in the folder"
        
        self.labels = loadmat(self.dataset_path + "/imagelabels.mat")['labels']
        # Squeeze to 1-D shape
        self.labels = torch.tensor(self.labels.astype(int)).squeeze(0)
        self.num_classes = self.labels.max()
        
        if split == 'train':
            self.idx = loadmat(self.dataset_path + "/setid.mat")['trnid']
        elif split == 'valid':
            self.idx = loadmat(self.dataset_path + "/setid.mat")['valid']
        else:
            self.idx = loadmat(self.dataset_path + "/setid.mat")['tstid']
            
        # Squeeze to (len) shape
        self.idx = torch.tensor(self.idx.astype(int)).squeeze(0)
        self.idx = torch.sort(self.idx).values
        
        if n_shot != -1:
            print(f"Sampled {n_shot}-shot samples from each of the classes")
            
            
        self.images = glob(self.dataset_path + "/jpg" + "/*.jpg")
        self.transform = transform
        
    def __getitem__(self, index):
        # Map the correct index to train and test index
        # Again the id is from 1 - 8189
        # We want 0 - 8188, so we minus 1
        idx = self.idx[index].item() - 1
        label = self.labels[idx]
        image = Image.open(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        
        # We want label to be from 0 - 101
        # The range of the label in the mat file is 1 - 102
        return image, label - 1
            
    def __len__(self):
        return len(self.idx)
        
        
        