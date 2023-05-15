# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import torch

from options import MonodepthOptions
from trainer_with_segm import TrainerWithSegm


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
    trainer = TrainerWithSegm(opts)
    trainer.train()
