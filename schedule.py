from scipy import interpolate
import torch
import numpy as np

if torch.cuda.is_available():
   device = 'cuda'
else:
   device = 'cpu'

def noising_sch(t, mode='exp_cubic', theta_start=1e3, theta_end = 1e-3, T=1000):
    device = t.device
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
