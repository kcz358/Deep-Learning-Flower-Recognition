from .base_loss import BaseLoss

from torch.nn import CrossEntropyLoss
from pytorch_metric_learning import losses, miners


class TripletLoss(BaseLoss):
    def __init__(self, 
                 loss_name: str,
                 offline : bool = False,
                 margin : float = 0.05,
                 swap : bool = False,
                 smooth_loss : bool = False,
                 triplets_per_anchor = 'all'):
        super().__init__(loss_name)
        
        self.triplet_loss = losses.TripletMarginLoss(margin=margin, 
                                                     swap = swap, 
                                                     smooth_loss=smooth_loss, 
                                                     triplets_per_anchor=triplets_per_anchor)
        
        
        if not offline:
            self.miner = miners.TripletMarginMiner()
        
    def forward(self, output, labels):
        # Output = (logits, embedding)
        hard_pairs = self.miner(output[1], labels)
        triplet_loss = self.triplet_loss(output[1], labels, hard_pairs)
        return triplet_loss