import torch
import math


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def bits_per_dim(nll: torch.Tensor, dim: int) -> torch.Tensor:
    # nll in nats -> bits, normalize by dimension
    return (nll / math.log(2)) / dim
