import torch

device = 'cpu'


def sample_t_batch(batch_size, T=1000):
    
    cat_dist = torch.distributions.categorical.Categorical(1 / T * torch.ones([T]))
    
    t_batch = cat_dist.sample([batch_size]).to(device) + 1
    
    return t_batch
