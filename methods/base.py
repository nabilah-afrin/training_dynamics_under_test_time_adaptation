import torch.nn as nn

class BaseTTA(nn.Module):
    """No adaptation. Wraps the base model for a uniform interface."""
    name = 'none'
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)
    def reset(self):
        pass