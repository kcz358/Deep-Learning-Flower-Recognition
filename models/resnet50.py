from typing import Iterator
from torch.nn.parameter import Parameter
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

default_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNet50(BaseModel):
    def __init__(self, 
                 model_name: str, 
                 num_classes : int,
                 weights : str= None, 
                 transformation = default_transformation):
        super().__init__(model_name)
        self.encoder = models.resnet50(weights=weights)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, num_classes)
        self.transformation = transformation
    
    def forward(self, x):
        return self.encoder(x)
    
    def to(self, device):
        self.encoder.to(device)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.encoder.parameters(recurse)
