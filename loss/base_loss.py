from typing import Dict
import importlib

import torch.nn as nn
AVAILABLE_LOSS = {
    'crossentropy' : 'CrossEntropy'
}

class BaseLoss(nn.Module):
    def __init__(self, loss_name : str):
        super().__init__()
        self.name = loss_name
    
def load_loss(loss_name: str, loss_args: Dict[str, str]) -> BaseLoss:
        assert loss_name in AVAILABLE_LOSS, f"{loss_name} is not an available loss function."
        module_path = f"loss.{loss_name}"
        imported_module = importlib.import_module(module_path)
        loss_formal_name = AVAILABLE_LOSS[loss_name]
        model_class = getattr(imported_module, loss_formal_name)
        print(f"Imported class: {model_class}")
        return model_class(**loss_args)