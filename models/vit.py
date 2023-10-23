from typing import Iterator
from torch.nn.parameter import Parameter
from torchvision import transforms, models
import torch.nn as nn
import torch

from .base_model import BaseModel

default_transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ViT(BaseModel):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 weights: str = None,
                 transformation=default_transformation):
        super().__init__(model_name)
        self.encoder = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=False)
        self.fc = nn.Linear(self.encoder.head.in_features, num_classes)
        self.transformation = transformation

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

    def to(self, device):
        self.encoder.to(device)
        self.fc.to(device)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return list(self.encoder.parameters(recurse)) + list(self.fc.parameters())