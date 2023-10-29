from typing import Dict
import importlib

import torch.nn as nn

AVAILABLE_POOLING = {
    'avgpooling' : 'AdaptiveAvgPool2d',
    'sumpooling' : 'AdaptiveSumPool2d',
    'gem' : 'GeM'
}

class BasePooling(nn.Module):
    def __init__(self, pooling_name : str) -> None:
        super().__init__()
        self.name = pooling_name
    
    def forward(self, x):
        pass
    
def load_pooling(pool_name: str) -> BasePooling:
        assert pool_name in AVAILABLE_POOLING, f"{pool_name} is not an available model."
        module_path = f"models.pooling.{pool_name}"
        imported_module = importlib.import_module(module_path)
        pool_formal_name = AVAILABLE_POOLING[pool_name]
        model_class = getattr(imported_module, pool_formal_name)
        print(f"Imported class: {model_class}")
        return model_class(pool_name)