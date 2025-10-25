import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class RealNVPNode(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden_size: int):
        super().__init__()
        dim = mask.numel()
        self.dim = dim
        self.register_buffer("mask", mask.float())

        self.s_func = nn.Sequential(
            nn.Linear(dim, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, dim)
        )
        self.t_func = nn.Sequential(
            nn.Linear(dim, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, dim)
        )

        # per-dimension learned scaling of s; start at 0 for stability
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (B, D)
        x_mask = x * self.mask
        raw_s = self.s_func(x_mask)
        # stabilize s
        s = torch.tanh(raw_s) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = ((1.0 - self.mask) * s).sum(dim=-1)  # (B,)
        return y, log_det

    def inverse(self, y):
        y_mask = y * self.mask
        raw_s = self.s_func(y_mask)
        s = torch.tanh(raw_s) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1.0 - self.mask) * (y - t) * torch.exp(-s)
        inv_log_det = ((1.0 - self.mask) * (-s)).sum(dim=-1)  # (B,)
        return x, inv_log_det


class RealNVP(nn.Module):
    def __init__(self, masks, hidden_size: int):
        super().__init__()
        self.dim = masks[0].numel()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([RealNVPNode(m, hidden_size) for m in masks])

    def _base(self, device):
        loc = torch.zeros(self.dim, device=device)
        cov = torch.eye(self.dim, device=device)
        return MultivariateNormal(loc, cov)

    def forward(self, x):
        """ Encode x -> z and accumulate log_det. Returns (z, log_det). """
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.layers:
            z, ld = layer.forward(z)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        """ Decode z -> x (exact inverse). Returns (x, inv_log_det). """
        inv_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        x = z
        for layer in reversed(self.layers):
            x, ild = layer.inverse(x)
            inv_log_det += ild
        return x, inv_log_det

    def log_probability(self, x):
        """ log p_X(x) = log p_Z(z) + log|det J| """
        z, log_det = self.forward(x)
        dist = self._base(x.device)
        return dist.log_prob(z) + log_det

    def sample(self, num_samples: int, device=None):
        device = device or next(self.parameters()).device
        dist = self._base(device)
        z = dist.sample((num_samples,))  # (B, D)
        x, _ = self.inverse(z)
        return x



def make_realnvp_masks(emb_dim: int, n_layers: int):
    """Alternating half-masks of size emb_dim."""
    masks = []
    half = emb_dim // 2
    m = torch.zeros(emb_dim)
    m[:half] = 1.0
    for _ in range(n_layers):
        masks.append(m.clone())
        m = 1.0 - m  # flip 0 <-> 1 for next layer
    return masks


"""
flow = RealNVP(make_realnvp_masks(1024, 6), hidden_size=512).to(device)
x = batch_embeddings.to(device)  # (B, 1024)
nll = -flow.log_probability(x).mean()
nll.backward()
optimizer.step()
"""
