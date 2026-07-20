"""
TENT adapter.
Reference: https://github.com/DequanWang/tent
"""
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from .base import BaseTTA


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing."""
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    outputs = model(x)
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


# ==========================================
# TENT Wrapper (Integrates with BaseTTA)
# ==========================================

class TentTTA(BaseTTA):
    name = 'tent'
    
    def __init__(self, base_model, lr=1e-3, **kwargs):
        super().__init__(base_model)
        
        # 1. configure_model (from your get_tent_model)
        wrapper = configure_model(base_model)
        
        # 2. collect_params (from your get_tent_model)
        params, _ = collect_params(wrapper)
        
        # 3. init Adam (from your get_tent_model)
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # 4. return tent.Tent (from your get_tent_model)
        self.tent = Tent(wrapper, optimizer)
        
        # Ensure the inner model is accessible for hooks
        self.model = self.tent.model

    def forward(self, x):
        return self.tent(x)

    def reset(self):
        self.tent.reset()