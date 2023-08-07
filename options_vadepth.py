# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import argparse
import os

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        self.parser.add_argument("-n", "--dry_run", action="store_true")
        self.parser.add_argument("--local", action="store_true", help="whether training on local GPU or on server")

        # PATHS
        self.parser.add_argument("--data_path", type=str, help="path to the training data", default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--root_log_dir", type=str, help="log directory", default=os.path.join(file_dir, "new_logs"))

        # SCALE_LOSS options
        self.parser.add_argument("--cam_height", type=float, help="camera height w.r.t road surface(m)", default=1.70)
        self.parser.add_argument("--left_rate", type=float, help="left_rate*w: horizontal coord of most-left point", default=0.425)
        self.parser.add_argument("--right_rate", type=float, help="left_rate*w: horizontal coord of most-right point", default=0.575)
        self.parser.add_argument("--upper_rate", type=float, help="left_rate*w: v-coord of most-upper point", default=0.875)
        self.parser.add_argument("--delta", type=float, help="threshold to determine whether a point is an inlier", default=1e-2)
        self.parser.add_argument("--height_loss_weight", type=float, help="scale loss weight for camera height", default=0)
        self.parser.add_argument("--disable_CPD", help="if set, doesn't do Co-planar Points Detection on Ground Surface", action="store_true")

        # TRAINING options
        self.parser.add_argument("--gamma", type=float, help="gamma for step_lr_scheduler", default=0.1)
        self.parser.add_argument("--random_seed", help="random seed", type=int)
        self.parser.add_argument(
            "--standard_metric",
            type=str,
            help="metric as an indicator of best model",
            choices=["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3", "loss"],
            default="loss",
        )

        self.parser.add_argument("--resume", help="whether resuming training", action="store_true")
        self.parser.add_argument("--ckpt_timestamp", type=str, help="this arg is valid only when specifying --resume")
        self.parser.add_argument("--last_epoch_for_resume", type=int, help="the last epoch number for resuming training")
        self.parser.add_argument("--model_name", type=str, help="the name of the folder to save the model in", default="")
        self.parser.add_argument(
            "--split",
            type=str,
            help="which training split to use",
            choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
            default="eigen_zhou",
        )
        self.parser.add_argument("--num_layers", type=int, help="number of resnet layers", default=18, choices=[18, 34, 50, 101, 152])
        self.parser.add_argument(
            "--dataset",
            type=str,
            help="dataset to train on",
            default="kitti",
            choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"],
        )
        self.parser.add_argument("--png", help="if set, trains from raw KITTI png files (instead of jpegs)", action="store_true")
        self.parser.add_argument("--height", type=int, help="input image height", default=192)
        self.parser.add_argument("--width", type=int, help="input image width", default=640)
        self.parser.add_argument("--disparity_smoothness", type=float, help="disparity smoothness weight", default=1e-3)
        self.parser.add_argument("--scales", nargs="+", type=int, help="scales used in the loss", default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)
        self.parser.add_argument("--max_depth", type=float, help="maximum depth", default=100.0)
        self.parser.add_argument("--use_stereo", help="if set, uses stereo pair for training", action="store_true")
        self.parser.add_argument(
            "--adj_frame_idxs",
            nargs="+",
            type=int,
            help="indices of adjacent frames centered on the frame of current interest",
            default=[0, -1, 1],
        )

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=12)
        self.parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-4)
        self.parser.add_argument("--num_epochs", type=int, help="number of epochs (including epochs of suspended training)", default=25)
        self.parser.add_argument("--scheduler_step_size", type=int, help="step size of the scheduler", default=15)

        # ABLATION options
        self.parser.add_argument("--avg_reprojection", help="if set, uses average reprojection loss", action="store_true")
        self.parser.add_argument("--disable_automasking", help="if set, doesn't do auto-masking", action="store_true")
        self.parser.add_argument("--predictive_mask", help="if set, uses a predictive masking scheme as in Zhou et al", action="store_true")
        self.parser.add_argument("--no_ssim", help="if set, disables ssim in the loss", action="store_true")
        self.parser.add_argument(
            "--weights_init",
            type=str,
            help="pretrained or scratch",
            default="pretrained",
            choices=["pretrained", "scratch"],
        )
        self.parser.add_argument(
            "--pose_model_input",
            type=str,
            help="how many images the pose network gets",
            default="pairs",
            choices=["pairs", "all"],
        )
        self.parser.add_argument(
            "--pose_model_type",
            type=str,
            help="normal or shared",
            default="separate_resnet",
            choices=["posecnn", "separate_resnet", "shared"],
        )

        # SYSTEM options
        self.parser.add_argument("--no_cuda", help="if set disables CUDA", action="store_true")
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=4)

        # LOADING options
        # self.parser.add_argument("--load_weights_folder", type=str, help="name of model to load")
        self.parser.add_argument(
            "--models_to_load",
            nargs="+",
            type=str,
            help="models to load",
            default=["encoder", "depth", "pose_encoder", "pose"],
        )

        # LOGGING options
        self.parser.add_argument("--log_frequency", type=int, help="number of batches between each tensorboard log", default=500)
        self.parser.add_argument("--log_image", action="store_true", help="whether saving disparities, automasks and images in log")
        ## save_frequency is invalid in trainer_with_segm.py
        self.parser.add_argument("--save_frequency", type=int, help="number of epochs between each save", default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo", help="if set evaluates in stereo mode", action="store_true")
        self.parser.add_argument("--eval_mono", help="if set evaluates in mono mode", action="store_true")
        self.parser.add_argument("--disable_median_scaling", help="if set disables median scaling in evaluation", action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor", help="if set multiplies predictions by this number", type=float, default=1)
        # self.parser.add_argument("--ext_disp_to_eval", type=str, help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--disp_filename_to_eval", type=str, help=".npy disparity filename to evaluate")
        self.parser.add_argument(
            "--eval_split",
            type=str,
            default="eigen",
            choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
            help="which split to run eval on",
        )
        self.parser.add_argument("--save_pred_disps", help="if set saves predicted disparities", action="store_true")
        self.parser.add_argument("--no_eval", help="if set disables evaluation", action="store_true")
        self.parser.add_argument(
            "--eval_eigen_to_benchmark",
            help="if set assume we are loading eigen results from npy but " "we want to evaluate using the new benchmark.",
            action="store_true",
        )
        self.parser.add_argument("--eval_out_dir", help="if set will output the disparities to this folder", type=str)
        self.parser.add_argument(
            "--post_process",
            help="if set will perform the flipping post processing " "from the original monodepth paper",
            action="store_true",
        )
        self.parser.add_argument("--epoch_for_eval", help="the number epochs for using evaluation", type=int)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
