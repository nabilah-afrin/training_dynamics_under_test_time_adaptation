from robustbench.utils import load_model

def get_base_model(cfg):
    model = load_model(model_name=cfg.model.name, dataset=cfg.model.dataset, threat_model=cfg.model.threat_model)
    return model.eval()


