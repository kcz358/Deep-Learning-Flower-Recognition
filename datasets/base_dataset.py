from typing import Dict
import importlib

from torch.utils.data import Dataset

AVAILABLE_DATASETS = {
    'flowers102' : 'Flowers102'
}

class BaseDataset(Dataset):
    def __init__(self, dataset_name : str) -> None:
        super().__init__()
        self.name = dataset_name
    
    def __getitem__(self, index):
        pass
    
    def __len__(self, index):
        pass

def load_dataset(dataset_name: str, dataset_args: Dict[str, str]) -> BaseDataset:
        assert dataset_name in AVAILABLE_DATASETS, f"{dataset_name} is not an available model."
        module_path = f"datasets.{dataset_name}"
        imported_module = importlib.import_module(module_path)
        dataset_formal_name = AVAILABLE_DATASETS[dataset_name]
        model_class = getattr(imported_module, dataset_formal_name)
        print(f"Imported class: {model_class}")
        return model_class(**dataset_args)