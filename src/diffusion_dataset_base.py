import torch
import sys
sys.path.append("..")

class ForwardDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, x_0_tensor, t_batch_tensor, suff_stats_tensor):
        super().__init__()
        self.x_0_tensor = x_0_tensor
        self.t_batch_tensor = t_batch_tensor
        self.suff_stats_tensor = suff_stats_tensor
        
    def __len__(self):
        return self.x_0_tensor.shape[0]
        
    def __getitem__(self, idx):
        return self.x_0_tensor[idx], self.t_batch_tensor[idx], self.suff_stats_tensor[idx]
    
    def stats_norm_fn(self):
        pass
    
def load_generated_dataset():
    x_0_tensor, t_batch_tensor, suff_stats_tensor = torch.load('/assets/x_0_dataset.pth'), torch.load('/assets/t_batch_dataset.pth'), torch.load('/assets/suff_stats_dataset.pth')
    return ForwardDiffusionDataset(x_0_tensor, t_batch_tensor, suff_stats_tensor)