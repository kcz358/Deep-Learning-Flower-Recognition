import torch.nn as nn

class AdaptiveSumPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool2d(output_size=(1,1))
    
    def forward(self, x):
        mu = x.mean()
        # avg * mu = sum
        return self.pooling(x) * mu