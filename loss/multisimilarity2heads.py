from .base_loss import BaseLoss

from torch.nn import CrossEntropyLoss
from pytorch_metric_learning import losses, miners


class Multisimilarity2Heads(BaseLoss):
    def __init__(self, 
                 loss_name: str,
                 offline : bool = False,
                 margin : float = 0.05,
                 swap : bool = False,
                 smooth_loss : bool = False,
                 triplets_per_anchor = 'all'):
        super().__init__(loss_name)
        
        self.triplet_loss = losses.MultipleLosses(margin=margin, 
                                                     swap = swap, 
                                                     smooth_loss=smooth_loss, 
                                                     triplets_per_anchor=triplets_per_anchor)
        
        self.cross_entropy = CrossEntropyLoss()
        
        if not offline:
            self.miner = miners.MultiSimilarityMiner()
        
    def forward(self, output, labels):
        # Output = (logits, embedding)
        # Scale the loss
        cls_loss = self.cross_entropy(output[0], labels) / 10
        hard_pairs = self.miner(output[1], labels)
        triplet_loss = self.triplet_loss(output[1], labels, hard_pairs)
        return cls_loss + triplet_loss