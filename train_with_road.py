# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from options import MonodepthOptions
from trainer_with_road import TrainerWithRoad
from utils import seed_all

options = MonodepthOptions()
opts = options.parse()
seed_all(opts.random_seed)


if __name__ == "__main__":
    trainer = TrainerWithRoad(opts)
    trainer.train()
