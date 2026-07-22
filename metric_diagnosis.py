import os, sys, numpy as np, torch

# Add root to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config
from utils.hooks import get_layer_names, register_hooks, remove_hooks
from utils.scenarios import resolve_scenarios
from data.dataloader import get_clean_loader, get_noisy_loader
from models.model_loader import get_base_model
from methods import build_method
from metrics import METRICS_MODULES

def _inner(m):
    return m.model if hasattr(m, 'model') else m

def _run_one(metric_mod, predict_fn, hook_model, loader, layer_names, device, is_adapt):
    hooks = register_hooks(hook_model, layer_names)
    arr = np.zeros((len(layer_names), len(loader)))
    for bidx, (x, _) in enumerate(loader):
        x = x.to(device)
        if is_adapt:
            _ = predict_fn(x)
        else:
            with torch.no_grad():
                _ = predict_fn(x)
        for li, name in enumerate(layer_names):
            H = metric_mod.flatten(hooks[name].activation)
            arr[li, bidx] = metric_mod.compute(H)
    remove_hooks(hooks)
    return arr.mean(axis=1)

def main(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device(cfg.device)
    metric_mod = METRIC_MODULES[cfg.metric]
    scenarios = resolve_scenarios(cfg)
    out_dir = f"results/rank_{cfg.dataset}_{cfg.model.name}_{cfg.method}_{cfg.metric}"
    os.makedirs(out_dir, exist_ok=True)

    tmp = get_base_model(cfg).to(device)
    layer_names = get_layer_names(tmp, cfg.layer_type)
    del tmp; torch.cuda.empty_cache()
    print('clean loader')
    clean_loader = get_clean_loader(cfg)

    print('noisy loaders')
    noisy_loaders = {sev: get_noisy_loader(cfg, sev) for sev in cfg.severities}

    if 'clean_base' in scenarios:
        print("[1] clean_base")
        m = get_base_model(cfg).to(device)
        avg = _run_one(metric_mod, m, m, clean_loader, layer_names, device, is_adapt=False)
        np.save(f"{out_dir}/clean_base.npy", avg)
        del m; torch.cuda.empty_cache()

    if 'noisy_base' in scenarios:
        print("[2] noisy_base")
        m = get_base_model(cfg).to(device)
        for sev in cfg.severities:
            avg = _run_one(metric_mod, m, m, noisy_loaders[sev], layer_names, device, is_adapt=False)
            np.save(f"{out_dir}/noisy_base_{cfg.dataset}_sev{sev}.npy", avg)
        del m; torch.cuda.empty_cache()

    if 'clean_tta' in scenarios:
        print("[3] clean_tta")
        base = get_base_model(cfg).to(device)
        method = build_method(cfg, base).to(device)
        avg = _run_one(metric_mod, method, _inner(method), clean_loader, layer_names, device, is_adapt=True)
        np.save(f"{out_dir}/clean_{cfg.dataset}_{cfg.method}.npy", avg)
        del method, base; torch.cuda.empty_cache()

    if 'noisy_tta' in scenarios:
        print("[4] noisy_tta")
        for sev in cfg.severities:
            base = get_base_model(cfg).to(device)
            method = build_method(cfg, base).to(device)
            avg = _run_one(metric_mod, method, _inner(method), noisy_loaders[sev], layer_names, device, is_adapt=True)
            np.save(f"{out_dir}/noisy_{cfg.dataset}_{cfg.method}_sev{sev}.npy", avg)
            del method, base; torch.cuda.empty_cache()

    print(f"Done. Results -> {out_dir}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")