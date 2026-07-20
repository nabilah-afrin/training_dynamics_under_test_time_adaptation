import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from robustbench.data import load_cifar10c, load_cifar100c

NORMALIZE = {
    'cifar10':   dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100':  dict(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    'imagenet':  dict(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225]),
}

def get_clean_loader(cfg):
    transform = T.Compose([T.ToTensor(), T.Normalize(**NORMALIZE[cfg.dataset])])
    if cfg.dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform)
    elif cfg.dataset == 'cifar100':
        ds = torchvision.datasets.CIFAR100(root=cfg.data_dir, train=False, download=True, transform=transform)
    elif cfg.dataset == 'imagenet':
        raise NotImplementedError("Add ImageNet validation loader.")
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

def get_noisy_loader(cfg, severity):
    if cfg.dataset == 'cifar10':
        x, _ = load_cifar10c(n_examples=cfg.n_examples, severity=severity, data_dir=cfg.data_dir, shuffle=False, corruptions=cfg.corruptions)
    elif cfg.dataset == 'cifar100':
        x, _ = load_cifar100c(n_examples=cfg.n_examples, severity=severity, data_dir=cfg.data_dir, shuffle=False, corruptions=cfg.corruptions)
    elif cfg.dataset == 'imagenet':
        raise NotImplementedError("Add ImageNet-C loader.")
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")
    return DataLoader(TensorDataset(x, torch.zeros(cfg.n_examples)), batch_size=cfg.batch_size, shuffle=False)