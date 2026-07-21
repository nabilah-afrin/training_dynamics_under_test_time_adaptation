import torch.nn as nn
from .base import BaseTTA
from .tent import TentTTA
from .cotta import CottaTTA

# Universal registry for all methods
METHODS = {
    'none': BaseTTA,
    'tent': TentTTA,
    'cotta': CottaTTA,
}

def build_method(cfg, base_model: nn.Module):

    cls = METHODS[cfg.method]
    kwargs = getattr(cfg, 'method_kwargs', {}) or {}
    
    # Pass the base model and config kwargs to the selected method class
    
    kwargs_dict = vars(kwargs) if not isinstance(kwargs, dict) else kwargs
    return cls(base_model, **kwargs_dict)