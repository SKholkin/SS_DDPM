from tqdm.auto import tqdm
from ..beta import KL
import torch
from src.diffusion_dataset_base import load_generated_dataset

import torch.nn.functional as F
from models import UNet

model = UNet(1, 32, (1, 2, 4), time_emb_dim=16)

class BetaUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(1, 32, (1, 2, 4), time_emb_dim=16)

    def forward(self, suff_stats, t):
        return F.sigmoid(self.model(suff_stats, t))


opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_generated_dataset(model, opt, dataloader, max_iter=1000):
    device = "cuda"
    pbar = tqdm(range(max_iter))
    model.train()
    
    for i in pbar:
        x_0, t_batch, suff_stats = next(iter(dataloader))
        x_0, t_batch, suff_stats = x_0.to(device), t_batch.to(device), suff_stats.to(device)
        
        x_theta = model(suff_stats, t_batch)
        
        loss = KL(x_0, x_theta, t_batch).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        pbar.set_description(f'Iter {i} Loss: {loss.item():.4f}')
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), f'beta_ddpm_{i}_iter.pth')
    return model

if __name__ == "__main__":
    model = BetaUnet()
    model = model.to("cuda")
    # model.load_state_dict(torch.load('beta_ddpm_the_best_100k_alpha_normed.pth'))
    model = UNet(1, 32, (1, 2, 4), time_emb_dim=16)
    forward_diffusion_dataset = load_generated_dataset()
    forward_diffusion_dataloader = torch.utils.data.DataLoader(forward_diffusion_dataset, batch_size=64, shuffle=True)
    trained_model = train_generated_dataset(model, opt, forward_diffusion_dataloader, max_iter=40000)
    
    print("Model trained, saving")
    torch.save(trained_model.state_dict(), "../assets/trained_model.pt")
    