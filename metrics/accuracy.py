import torch

def evaluate_accuracy(predict_fn, loader, device, is_adapt=False):
    """Adaptation-aware: TTA methods need gradients, so we don't no_grad them."""
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_adapt:
            out = predict_fn(x)
        else:
            with torch.no_grad():
                out = predict_fn(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)