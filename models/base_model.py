import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, model_name : str):
        super().__init__()
        self.name = model_name
    
    def to(self, device):
        pass