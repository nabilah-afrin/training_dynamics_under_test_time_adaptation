import torch

# SVD rank computation
def compute_energy_rank(feat_tensor, thresh):
    feat = feat_tensor.view(feat_tensor.size(1), -1)
    _, s, _ = torch.linalg.svd(feat, full_matrices=False)
    energy = torch.cumsum(s**2, dim=0)
    total = energy[-1]
    idx = torch.searchsorted(energy, total * thresh)
    return (idx + 1).item()