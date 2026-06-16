"""Random helper stuff I use everywhere - seeding, metrics, saving, label names."""
import os
import random

import numpy as np
import torch

# the 7 emotions in the same order as the labels in the csv (0=Angry, ..., 6=Neutral)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTIONS)


def set_seed(seed: int = 42):
    """Seed everything so runs are mostly repeatable (gpu is never 100% deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark mode picks fast kernels - a bit less deterministic but faster, don't care here
    torch.backends.cudnn.benchmark = True


def count_parameters(model: torch.nn.Module) -> int:
    """how many trainable params the model has."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """just keeps a running average, handy for loss/acc over a bunch of batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
