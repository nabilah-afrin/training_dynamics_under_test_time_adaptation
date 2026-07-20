import os, sys, json, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config
from utils.scenarios import resolve_scenarios
from data.datasets import get_clean_loader, get_noisy_loader
from models.model_loader import get_base_model
from methods import build_method
from metrics.accuracy import evaluate_accuracy

def _inner(m):
    return m.model if hasattr(m, 'model') else m

def main(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device(cfg.device)
    scenarios = resolve_scenarios(cfg)
    out_dir = f"results/acc_{cfg.dataset}_{cfg.model.name}_{cfg.method}"
    os.makedirs(out_dir, exist_ok=True)

    clean_loader = get_clean_loader(cfg)
    noisy_loaders = {sev: get_noisy_loader(cfg, sev) for sev in cfg.severities}

    results = {}

    if 'clean_base' in scenarios:
        m = get_base_model(cfg).to(device)
        results['clean_base'] = evaluate_accuracy(m, clean_loader, device, is_adapt=False)
        del m; torch.cuda.empty_cache()

    if 'noisy_base' in scenarios:
        m = get_base_model(cfg).to(device)
        for sev in cfg.severities:
            results[f'noisy_base_sev{sev}'] = evaluate_accuracy(m, noisy_loaders[sev], device, is_adapt=False)
        del m; torch.cuda.empty_cache()

    if 'clean_tta' in scenarios:
        base = get_base_model(cfg).to(device)
        method = build_method(cfg, base).to(device)
        results['clean_tta'] = evaluate_accuracy(method, clean_loader, device, is_adapt=True)
        del method, base; torch.cuda.empty_cache()

    if 'noisy_tta' in scenarios:
        for sev in cfg.severities:
            base = get_base_model(cfg).to(device)
            method = build_method(cfg, base).to(device)
            results[f'noisy_tta_sev{sev}'] = evaluate_accuracy(method, noisy_loaders[sev], device, is_adapt=True)
            del method, base; torch.cuda.empty_cache()

    print("=== Accuracy ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_dir}/results.json")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")