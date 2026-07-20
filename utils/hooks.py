import torch.nn as nn

class LayerActivationHook:
    def __init__(self):
        self.activation = None
        self._handle = None
    def register(self, layer):
        self._handle = layer.register_forward_hook(self._fn)
        return self
    def _fn(self, module, inp, out):
        self.activation = out.detach()
    def remove(self):
        if self._handle:
            self._handle.remove()

def get_layer_names(model, layer_type='Conv2d'):
    cls = getattr(nn, layer_type)
    return [name for name, m in model.named_modules() if isinstance(m, cls)]

def register_hooks(model, layer_names):
    mods = dict(model.named_modules())
    return {n: LayerActivationHook().register(mods[n]) for n in layer_names}

def remove_hooks(hooks):
    for h in hooks.values():
        h.remove()

def flatten_activation(act):
    C = act.shape[1]
    return act.permute(1, 0, 2, 3).reshape(C, -1)