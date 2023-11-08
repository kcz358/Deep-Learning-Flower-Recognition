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
                 feature_extraction = False,
                 ):
        super().__init__(model_name)
        self.encoder = models.vit_b_16(weights = weights, progress = True, dropout = dropout_rate)
        self.encoder.heads[0] = nn.Linear(self.encoder.heads[0].in_features, num_classes) # originally (head): Linear(in_features=768, out_features=1000, bias=True)
        self.transformation = transformation
        self.feature_extraction = feature_extraction
        self.embedding_size = 768
        self.cnt = 0 # aux variable
        print(self.encoder)
        print(f"transformation: {self.transformation}")
        print(f"feature extraction: {self.feature_extraction}")

    def forward(self, x):
        N, C, H, W = x.shape

        if self.feature_extraction:
            if self.cnt == 0:
                y = self.encoder.conv_proj(x)
                print(y.shape)
                y = y.reshape(N, self.encoder.hidden_dim, -1)
                print(y.shape)
                y = y.permute(0, 2, 1)
                bct = self.encoder.class_token.expand(N, -1, -1)
                y = torch.cat([bct, y], dim = 1)
                print(y.shape)
                embedding = self.encoder.encoder(y)[:, 0, :]
                print(embedding.shape)
                self.cnt += 1
            else:
                y = self.encoder.conv_proj(x)
                y = y.reshape(N, self.encoder.hidden_dim, -1)
                y = y.permute(0, 2, 1)
                bct = self.encoder.class_token.expand(N, -1, -1)
                y = torch.cat([bct, y], dim = 1)
                embedding = self.encoder.encoder(y)[:, 0, :]
        else:
            embedding = None
        return self.encoder(x), embedding # no feature extraction yet

    def to(self, device):
        self.encoder.to(device)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.encoder.parameters(recurse)
    
    def registerHook(self):
        self.attention_map = None
        def hook(model, input, output):
            self.attention_map = output.detach()
        self.encoder.encoder.layers[-1].self_attention.register_forward_hook(hook)


