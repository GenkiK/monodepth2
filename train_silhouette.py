from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import torch

from options import MonodepthOptions
from trainer_silhouette import TrainerSilhouette


def seed_all(seed):
    if not seed:
        seed = 1
    print(f"Using seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True  # TODO: change to False for reproducibility


options = MonodepthOptions()
opts = options.parse()
seed_all(opts.random_seed)


if __name__ == "__main__":
    trainer = TrainerSilhouette(opts)
    trainer.train()
