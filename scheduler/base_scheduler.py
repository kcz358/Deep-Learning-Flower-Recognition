from typing import Dict
import importlib
from abc import ABC, abstractmethod

AVAILABLE_SCHEDULER = {
    'steplr' : 'StepScheduler'
}

class BaseScheduler(ABC):
    def __init__(self, scheduler_name : str, optimizer):
        self.name = scheduler_name
        self.optimizer = optimizer
    
    @abstractmethod
    def get_last_lr(self):
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    
    @abstractmethod
    def print_lr(self):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass
    
    @abstractmethod
    def step(self):
        pass
    
    
def load_scheduler(scheduler_name: str, scheduler_args: Dict[str, str]) -> BaseScheduler:
        assert scheduler_name in AVAILABLE_SCHEDULER, f"{scheduler_name} is not an available scheduler."
        module_path = f"scheduler.{scheduler_name}"
        imported_module = importlib.import_module(module_path)
        scheduler_formal_name = AVAILABLE_SCHEDULER[scheduler_name]
        model_class = getattr(imported_module, scheduler_formal_name)
        print(f"Imported class: {model_class}")
        return model_class(**scheduler_args)