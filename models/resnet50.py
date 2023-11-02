from typing import Iterator
import itertools

from torch.nn.parameter import Parameter
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .pooling.base_pooling import load_pooling

default_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tuned_option = {
    'full' : 0,
    #Freeze everything before the first layer
    'low' : 4,
    #Freeze everything before the third layer
    'mid' : 6,
    #Freeze everything before layer 4
    'high' : 7,
    # Only tuned the linear head
    'linear_head' : -2
}

class ResNet50(BaseModel):
    def __init__(self, 
                 model_name: str, 
                 num_classes : int,
                 weights : str= None, 
                 transformation = default_transformation,
                 feature_extraction : bool = False,
                 tuned : str = 'full',
                 pooling : str = 'avgpooling'):
        super().__init__(model_name)
        assert tuned.lower() in tuned_option, "Specify the portion you want to tuned. Available option [full, low, mid, high, linear_head]"
        model = models.resnet50(weights = weights)
        layers = list(model.children())[:-2]
        self.off_layers = tuned_option[tuned.lower()]
        for l in layers[:self.off_layers]:
            for p in l.parameters():
                p.requires_grad = False
        

        self.encoder = nn.Sequential(*layers)
        self.pooling = load_pooling(pool_name=pooling)
        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024,2048)
        )
        self.embedding_size = 2048
        self.cls_head = nn.Sequential(
            nn.Linear(model.fc.in_features, num_classes),
            nn.Softmax(dim = 1)
        )
        self.transformation = transformation
        self.feature_extraction = feature_extraction
        
        del model
    
    def forward(self, x):
        features = self.encoder(x)
        # (B, 2048, 7, 7) --> (B, 2048, 1, 1) --> (B, 2048)
        features = self.pooling(features).flatten(1)
        embedding = self.embedding_head(features)
        output = self.cls_head(features)
        if self.feature_extraction:
            return output, embedding
        else:      
            return output, None
    
    def to(self, device):
        self.encoder.to(device)
        self.cls_head.to(device)
        self.embedding_head.to(device)
        # Gem is a trainable pooling layer
        if self.pooling.name != 'avgpooling' or self.pooling.name != 'sumpooling':
            self.pooling.to(device)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.pooling.name != 'avgpooling' or self.pooling.name != 'sumpooling':
            return itertools.chain(self.encoder.parameters(recurse), 
                               self.cls_head.parameters(recurse),
                               self.pooling.parameters(recurse),
                               self.embedding_head.parameters(recurse))

        return itertools.chain(self.encoder.parameters(recurse), 
                               self.cls_head.parameters(recurse),
                               self.embedding_head.parameters(recurse))
