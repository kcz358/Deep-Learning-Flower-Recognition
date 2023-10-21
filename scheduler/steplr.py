from torch.optim.lr_scheduler import StepLR

from .base_scheduler import BaseScheduler

class StepScheduler(BaseScheduler):
    def __init__(self, 
                 scheduler_name: str, 
                 optimizer, 
                 gamma : float, 
                 step_size : int, 
                 last_epoch=-1, 
                 verbose=False):
        super().__init__(scheduler_name, optimizer)
        self.scheduler = StepLR(optimizer.optimizer, 
                                gamma=gamma, 
                                step_size=step_size, 
                                last_epoch=last_epoch, 
                                verbose=verbose)
        
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)
    
    def print_lr(self):
        return self.scheduler.print_lr()
    
    def step(self):
        self.scheduler.step()
        
    def state_dict(self):
        return self.scheduler.state_dict()