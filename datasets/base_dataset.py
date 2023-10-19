from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dataset_name : str) -> None:
        super().__init__()
        self.name = dataset_name
    
    def __getitem__(self, index):
        pass
    
    def __len__(self, index):
        pass
    