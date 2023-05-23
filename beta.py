import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import torchvision
from scipy import interpolate
from torch.distributions.beta import Beta

from schedule import noising_sch

import numpy as np

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
    mu = noising_sch(t).reshape([-1, 1, 1, 1])
    
    return mu * torch.log(x_t / (1 - x_t))

def sufficient_stats_part2(x_t):
    
    return torch.log(x_t / (1 - x_t))
                        
def sample_chain(t_batch, x_0, T=1000):
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

def sample_chain_suff_stats_norm_alpha(t_batch, x_0, T=1000):
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

def get_dist(mu, x_0):
    alpha, beta = alpha_beta(mu, x_0)
    return Beta(alpha, beta)

