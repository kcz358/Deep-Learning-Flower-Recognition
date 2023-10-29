import torch.nn as nn

class AdaptiveAvgPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
    
    def forward(self, x):
        return self.pooling(x)