from .base_loss import BaseLoss

from pytorch_metric_learning import losses, miners


class MultiSimilarityLoss(BaseLoss):
    def __init__(self, 
                 loss_name: str,
                 offline : bool = False,
                 alpha=2, 
                 beta=50, 
                 base=0.5):
        super().__init__(loss_name)
        
        self.ms_loss = losses.MultiSimilarityLoss(alpha=alpha, beta=beta,base=base)
        
        
        if not offline:
            self.miner = miners.MultiSimilarityMiner()
        
    def forward(self, output, labels):
        # Output = (logits, embedding)
        hard_pairs = self.miner(output[1], labels)
        ms_loss = self.ms_loss(output[1], labels, hard_pairs)
        return ms_loss