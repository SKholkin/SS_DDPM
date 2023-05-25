# Star-Shaped DDPM Implementation
Repository for Skoltech Deep Learning Final Project. Reproduction of Star-Shaped Denoising Diffusion Probabilistic Models (Okhotin et. al https://arxiv.org/pdf/2302.05259.pdf) in PyTorch. 

## Motivation
From paper, `For data distributed on manifolds, bounded volumes, or with other features, the injection of Gaussian noise can be unnatural, breaking the data structure`. Following this observation, authors propose to use non-Gaussian noising scheme for constrained manifolds (e.g sphere), such as Mises-Fisher, Dirichlet, Beta etc.

## Generating Dataset samples in Offline Regime
Since in SSDDPM forward process can not be implemented in closed form (as for example in casual DDPM), it is more efficient to prepare in offline. In order to obtain such dataset, run (you can change hyper parameters in file)
```
python generate_dataset.py
```

## Notebooks
Some research code & dirty notebooks can be found in `playground` folder. UNet model is located in `models.py` and has nothing special in it.