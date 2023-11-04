import torch.nn as nn

from .base_pooling import BasePooling

class AdaptiveSumPool2d(BasePooling):
    def __init__(self, pooling_name: str) -> None:
        super().__init__(pooling_name)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
    
    def forward(self, x):
        mu = x.mean()
        # avg * mu = sum
        return self.pooling(x) * mu