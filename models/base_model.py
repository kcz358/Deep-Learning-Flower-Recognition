from typing import Dict
import importlib

import torch.nn as nn
AVAILABLE_MODELS = {
    'resnet50' : 'ResNet50'
}

class BaseModel(nn.Module):
    def __init__(self, model_name : str):
        super().__init__()
        self.name = model_name
    
    def to(self, device):
        pass
    
def load_model(model_name: str, model_args: Dict[str, str]) -> BaseModel:
        assert model_name in AVAILABLE_MODELS, f"{model_name} is not an available model."
        module_path = f"models.{model_name}"
        imported_module = importlib.import_module(module_path)
        model_formal_name = AVAILABLE_MODELS[model_name]
        model_class = getattr(imported_module, model_formal_name)
        print(f"Imported class: {model_class}")
        return model_class(**model_args)