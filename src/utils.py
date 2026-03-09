"""Utilidades: seeds, logging, I/O de configs e resultados."""

import os
import random
import logging
from pathlib import Path

import yaml
import numpy as np


def load_config(path: str="/home/tec/Projects/slm-spyware-detection/configs/experiment.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
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
