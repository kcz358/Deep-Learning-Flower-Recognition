from typing import Dict
import importlib
from abc import ABC, abstractmethod

AVAILABLE_OPTIMIZER = {
    'adam' : 'AdamOpt'
}

class BaseOptimizer(ABC):
    def __init__(self, opt_name : str, optimizer, warmup : int = 0):
        self.name = opt_name
        self.warmup = warmup
        self.optimizer = optimizer
    
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    
    
def load_optimizer(optimizer_name: str, optimizer_args: Dict[str, str]) -> BaseOptimizer:
        assert optimizer_name in AVAILABLE_OPTIMIZER, f"{optimizer_name} is not an available model."
        module_path = f"optimizer.{optimizer_name}"
        imported_module = importlib.import_module(module_path)
        optimizer_formal_name = AVAILABLE_OPTIMIZER[optimizer_name]
        model_class = getattr(imported_module, optimizer_formal_name)
        print(f"Imported class: {model_class}")
        return model_class(**optimizer_args)