from typing import Iterator
from torch.nn.parameter import Parameter
from torchvision import transforms, models
import torch.nn as nn
import torch

from .base_model import BaseModel

default_transformation = transforms.Compose([ # (conv_proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ViT(BaseModel):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 dropout_rate: int,
                 weights: str = None,
                 transformation=default_transformation,
                 ):
        super().__init__(model_name)
        self.encoder = models.vit_b_16(weights = weights, progress = True, dropout = dropout_rate)
        self.encoder.heads[0] = nn.Linear(self.encoder.heads[0].in_features, num_classes) # originally (head): Linear(in_features=768, out_features=1000, bias=True)
        self.transformation = transformation
        print(self.encoder)

    def forward(self, x):
        return self.encoder(x), None # no feature extraction yet

    def to(self, device):
        self.encoder.to(device)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.encoder.parameters(recurse)
