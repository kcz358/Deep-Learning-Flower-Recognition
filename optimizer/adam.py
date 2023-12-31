from .base_optimizer import BaseOptimizer
from .sam import SAM

from torch.optim import Adam

class AdamOpt(BaseOptimizer):
    def __init__(self, 
                 opt_name: str, 
                 param_groups, 
                 lr : float, 
                 betas=(0.9, 0.98),
                 weight_decay : float = 0, 
                 eps=1e-9, 
                 optimizer = Adam, 
                 warmup: int = 0,
                 sam : bool = False):
        super().__init__(opt_name, optimizer, warmup)
        self.optimizer = self.optimizer(param_groups, lr=lr, betas=betas, weight_decay = weight_decay, eps=eps)
        self._step = 0
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas=betas
        self.sam = sam
        print(f"SAM status: {self.sam}")
        if self.sam:
            self.optimizer = SAM(self.optimizer.param_groups, self.optimizer, adaptive=True)
        
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict) 
    
    def step(self, closure=None):
        if(self._step < self.warmup):
            lr_scale = min(1., float(self._step + 1) / self.warmup)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        
        self._step += 1
        if self.sam:
            self.optimizer.step(closure=closure)
        else:     
            self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()