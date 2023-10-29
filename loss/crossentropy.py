from .base_loss import BaseLoss

import torch.nn as nn

class CrossEntropy(BaseLoss):
    def __init__(self, 
                 loss_name: str,
                 weight = None,
                 size_average = None,
                 ignore_index = -100,
                 reduce=None,
                 reduction='mean',
                 label_smoothing=0.0):
        super().__init__(loss_name)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self, output, labels):
        # Contains encoding features
        # (logits, encoding features), will be None encoding features 
        # if feature extraction for resnet if false
        return self.criterion(output[0], labels)