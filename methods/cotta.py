"""
CoTTA adapter.
Reference: https://github.com/qinenergy/cotta
"""
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from .base import BaseTTA


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class Cotta(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing."""
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = self.model(x)
        
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        
        # Since we use no internal transforms across methods, 
        # the augmentation-averaged prediction is simply the standard EMA output.
        if anchor_prob.mean(0) < self.ap:
            outputs_ema = standard_ema
        else:
            outputs_ema = standard_ema
            
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
        
        # Stochastic restore
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.rst).float().to(p.device) 
                    with torch.no_grad():
                        p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
                        
        return outputs_ema


@torch.jit.script
def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters (BatchNorm only)."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent/cotta."""
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
# CoTTA Wrapper (Integrates with BaseTTA)
# ==========================================

class CottaTTA(BaseTTA):
    name = 'cotta'
    
    def __init__(self, base_model, lr=1e-3, mt_alpha=0.99, rst_m=0.1, ap=0.9, steps=1, episodic=False, **kwargs):
        super().__init__(base_model)
        
        # 1. Configure model for BN adaptation
        wrapper = configure_model(base_model)
        
        # 2. Collect parameters to update
        params, _ = collect_params(wrapper)
        
        # 3. Init Adam Optimizer
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # 4. Initialize Cotta engine
        self.cotta = Cotta(
            wrapper, optimizer, 
            steps=steps, episodic=episodic, 
            mt_alpha=mt_alpha, rst_m=rst_m, ap=ap
        )
        
        # Ensure the inner model is accessible for hooks
        self.model = self.cotta.model

    def forward(self, x):
        return self.cotta(x)

    def reset(self):
        self.cotta.reset()