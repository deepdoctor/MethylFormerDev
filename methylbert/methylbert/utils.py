
import math
import random
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import yaml
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path: str, overrides: Dict[str, Any] = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or {}
    def set_by_path(d, path, value):
        keys = path.split(".")
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    for k, v in overrides.items():
        set_by_path(cfg, k, v)
    return cfg

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max_steps if max_steps > 0 else None
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            elif self.max_steps is None:
                lr = base_lr
            else:
                # cosine decay from 1 to 0
                progress = min(1.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
