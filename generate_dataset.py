import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import torch
from torch.distributions.beta import Beta
from scipy import interpolate


batch_size = 2048
T = 1000
import numpy as np
device = 'cuda'

def sample_t_batch(batch_size):
    
    cat_dist = torch.distributions.categorical.Categorical(1 / T * torch.ones([T]))
    
    t_batch = cat_dist.sample([batch_size]).to(device) + 1
    
    return t_batch

def log_beta_fn(z_1, z_2):
    return torch.special.gammaln(z_1) + torch.special.gammaln(z_2) - torch.special.gammaln(z_1 + z_2)

def alpha_beta(mu, x_0):
    return 1 + mu.reshape([-1, 1, 1, 1]) * x_0, 1 + mu.reshape([-1, 1, 1, 1]) * (1 - x_0)

def KL(x_0, x_theta, t):
    mu = noising_sch(t).reshape([-1, 1, 1, 1]) 
    alpha_0, beta_0 = alpha_beta(mu, x_0)
    alpha_theta, beta_theta = alpha_beta(mu, x_theta)
    kl_div = log_beta_fn(alpha_theta, beta_theta)
    kl_div = kl_div - log_beta_fn(alpha_0, beta_0)
    kl_div = kl_div + mu * (x_0 - x_theta) * ((torch.special.digamma(alpha_0) - torch.special.digamma(beta_0)))
    return kl_div

def sufficient_stats(x_t, t):
    theta = noising_sch(t).reshape([-1, 1, 1, 1])
    
    return theta * torch.log(x_t / (1 - x_t))
                                                 

def noising_sch(t, mode='exp_cubic', theta_start=1e3, theta_end = 1e-3):
    if mode == 'linear':
        theta = theta_end + (T - t) / T * (theta_start - theta_end)
    elif mode=='exp_linear': 
        log10_theta = np.log10(theta_end) + (T - t) / T * (np.log10(theta_start) - np.log10(theta_end))
        theta = torch.pow(10, log10_theta)
    elif mode=='exp_cubic':
        spline = interpolate.CubicSpline([1, T * 0.3, T * 0.7, T], [3, 0.7, 0, -3])
        log10_theta = torch.Tensor(spline(t.cpu().numpy()))
        theta = torch.pow(10, log10_theta)
    else:
        raise BaseException('Unknown schedule mode')
        
    return torch.Tensor(theta).to(device)


def sample_chain_suff_stats_norm_alpha(t_batch, x_0):
    samples = []
    suff_stats = torch.zeros_like(x_0)
    suff_stats_normed = torch.zeros_like(x_0)
    t_min = torch.min(t_batch)
    helper = torch.Tensor([[t <= s for s in range(T + 1)] for t in t_batch]).to(device)
    
    alphas = torch.zeros_like(x_0)
    
    for s in range(T, int(t_min), -1):
        s_batch = torch.tensor([s], device=device).repeat(batch_size)
        helper_slice = helper[:, s]
        
        mu = noising_sch(s_batch)
        
        alpha, beta = alpha_beta(mu, x_0)
        
        dist = Beta(alpha, beta)
        samples.append(dist.sample())
        alphas += helper_slice.reshape([-1, 1, 1, 1]) * mu.reshape([-1, 1, 1, 1]).repeat([1, 1, 28, 28])
        
        suff_stats += helper_slice.reshape([-1, 1, 1, 1]) * sufficient_stats(samples[-1], s_batch)
        
    suff_stats_normed = suff_stats / alphas
        
    return samples, suff_stats_normed

def sample_chain(t_batch, x_0):
    samples = []
    suff_stats = torch.zeros_like(x_0)
    t_min = torch.min(t_batch)
    helper = torch.Tensor([[t <= s for s in range(T + 1)] for t in t_batch]).to(device)
    
    for s in range(T, int(t_min), -1):
        helper_slice = helper[:, s]
        
        theta = noising_sch(t_batch)
        
        alpha, beta = alpha_beta(theta, x_0)
        
        dist = Beta(alpha, beta)
        samples.append(dist.sample())
        s_batch = torch.tensor([s], device=device).repeat(batch_size)

        suff_stats += helper_slice.reshape([-1, 1, 1, 1]) * sufficient_stats(samples[-1], s_batch)
    return samples, suff_stats


def generate_dataset(pic_dataloader, n_samples, batch_size=1024, save_path='generated_dataset.pth'):
    n_iters = n_samples // batch_size
    
    x_0_storage = []
    t_batch_storage = []
    suff_stats_storage = []
    
    for i in tqdm(range(n_iters)):
        x_0 = next(iter(pic_dataloader))[0].to(device)
        t_batch = sample_t_batch(batch_size)
        samples, suff_stats = sample_chain_suff_stats_norm_alpha(t_batch, x_0)
        
        x_0_storage.append(x_0.cpu())
        t_batch_storage.append(t_batch.cpu())
        suff_stats_storage.append(suff_stats.cpu())
        
    x_0_tensor = torch.cat(x_0_storage, dim=0)
    t_batch_tensor = torch.cat(t_batch_storage, dim=0)
    suff_stats_tensor = torch.cat(suff_stats_storage, dim=0)
    return x_0_tensor, t_batch_tensor, suff_stats_tensor
        
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

dataset = MNIST(root='MNIST', download=True,transform=transforms.Compose([transforms.ToTensor()]))

import matplotlib.pyplot as plt

plt.imshow(dataset[0][0].squeeze(), cmap='gray')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
n_samples = 100000
x_0_tensor, t_batch_tensor, suff_stats_tensor = generate_dataset(dataloader, n_samples, batch_size=batch_size)

torch.save(x_0_tensor, 'x_0_dataset.pth')
torch.save(t_batch_tensor, 't_batch_dataset.pth')
torch.save(suff_stats_tensor, 'suff_stats_dataset.pth')
