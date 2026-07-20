import yaml
from types import SimpleNamespace

def _to_ns(o):
    if isinstance(o, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_ns(v) for v in o]
    return o

def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))