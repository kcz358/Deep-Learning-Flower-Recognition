from .base_loss import BaseLoss

from torch.nn import CrossEntropyLoss
from pytorch_metric_learning import losses, miners


class Multisimilarity2Heads(BaseLoss):
    def __init__(self, 
                 loss_name: str,
                 offline : bool = False,
                 alpha=2, 
                 beta=50, 
                 base=0.5):
        super().__init__(loss_name)
        
        self.ms_loss = losses.MultipleLosses(alpha=alpha, beta=beta,base=base)
        
        self.cross_entropy = CrossEntropyLoss()
        
        if not offline:
            self.miner = miners.MultiSimilarityMiner()
        
    def forward(self, output, labels):
        # Output = (logits, embedding)
        # Scale the loss
        cls_loss = self.cross_entropy(output[0], labels)
        hard_pairs = self.miner(output[1], labels)
        ms_loss = self.ms_loss(output[1], labels, hard_pairs)
        return cls_loss + ms_loss