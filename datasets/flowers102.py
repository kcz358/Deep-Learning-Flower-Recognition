from glob import glob
import os

from scipy.io import loadmat
import torch
from torchvision import transforms
from PIL import Image

from .base_dataset import BaseDataset

class Flowers102(BaseDataset):
    def __init__(self, dataset_path : str, split : str = 'train', transform = None) -> None:
        super().__init__("flower102")
        self.dataset_path = dataset_path
        assert os.path.exists(dataset_path + "/jpg"), "No jpg (image) files in the folder"
        assert os.path.exists(dataset_path + "/imagelabels.mat"), "No label files in the folder"
        assert os.path.exists(dataset_path + "/setid.mat"), "No dataset split files in the folder"
        
        self.labels = loadmat(self.dataset_path + "/imagelabels.mat")['labels']
        # Squeeze to 1-D shape
        self.labels = torch.tensor(self.labels.astype(int)).squeeze(0)
        self.num_classes = 102
        
        if split == 'train':
            self.idx = loadmat(self.dataset_path + "/setid.mat")['trnid']
        elif split == 'valid':
            self.idx = loadmat(self.dataset_path + "/setid.mat")['valid']
        else:
            self.idx = loadmat(self.dataset_path + "/setid.mat")['tstid']
            
        # Squeeze to (len) shape
        self.idx = torch.tensor(self.idx.astype(int)).squeeze(0)
        self.images = glob(self.dataset_path + "/jpg" + "/*.jpg")
        self.transform = transform
        
    def __getitem__(self, index):
        label = self.labels[index]
        image = Image.open(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
            
    def __len__(self):
        return len(self.idx)
        
        
        