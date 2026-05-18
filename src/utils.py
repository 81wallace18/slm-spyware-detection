"""Utilidades: seeds, logging, I/O de configs e resultados."""

import os
import random
import logging
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base and return base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path = DEFAULT_CONFIG, profile: str | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if profile:
        profile_path = REPO_ROOT / "configs" / "profiles" / f"{profile}.yaml"
        with open(profile_path) as f:
            cfg = deep_update(cfg, yaml.safe_load(f))
        cfg["active_profile"] = profile

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dirs(cfg: dict):
    for key, p in cfg.get("paths", {}).items():
        Path(p).mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str = "cuda") -> str:
    """Return a usable torch device, falling back to CPU when needed."""
    if requested != "cuda":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cleanup_torch_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_results(results: dict, path: str):
    """Salva dict de resultados em CSV ou JSON conforme extensão."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        pd.DataFrame(results).to_csv(path, index=False)
    else:
        import json
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
