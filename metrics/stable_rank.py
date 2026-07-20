import torch
from utils.hooks import flatten_activation

def safe_svdvals(H: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.svdvals(H)
    except Exception:
        return torch.linalg.svdvals(H.cpu()).to(H.device)

def compute(H: torch.Tensor) -> float:
    H = H.to(torch.float64)
    s = safe_svdvals(H)
    s_sq = s ** 2
    sum_sq = torch.sum(s_sq)
    sum_quad = torch.sum(s_sq ** 2)
    if sum_quad == 0:
        return 0.0
    return (sum_sq ** 2 / sum_quad).item()

# Expose a single flattening convention per metric (so eigen_value can differ later)
flatten = flatten_activation