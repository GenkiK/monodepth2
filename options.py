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

        # TRAINING options
        self.parser.add_argument(
            "--from_ground", action="store_true", help="when calculating scale factor, define silhouette height as from ground to the top"
        )
        self.parser.add_argument("--with_normal_loss", action="store_true")
        self.parser.add_argument("--gradual_normal_weight", action="store_true")
        self.parser.add_argument(
            "--normal_with_8_neighbors", action="store_true", help="when computing road normal, using 8-neighbors method, not pinverse"
        )
        self.parser.add_argument("--disable_road_masking", action="store_true", help="disable automasking only in the road region")
        self.parser.add_argument("--use_median_cam_height", action="store_true", help="use median camera height as a representative value")
        self.parser.add_argument("--use_median_scale", action="store_true", help="use median scale factor as a representative value")
        self.parser.add_argument("--soft_remove_outliers", action="store_true")
        self.parser.add_argument("--remove_outliers", action="store_true")
        self.parser.add_argument("--outlier_relative_error_th", type=float, default=0.2)
        self.parser.add_argument(
            "--use_median_depth", action="store_true", help="use median depth over instances as an representative depth for calculating scale factor"
        )
        self.parser.add_argument(
            "--use_1st_quartile_depth",
            action="store_true",
            help="use 1st quartile depth over instances as an representative depth for calculating scale factor",
        )
        self.parser.add_argument("--enable_erosion", action="store_true", help="enable eroding object segms when computing camera height")
        self.parser.add_argument("--init_after_1st_epoch", action="store_true")
        self.parser.add_argument("--log_dirname_1st_epoch", type=str, help="this argument is valid only when --init_after_1st_epoch")
        self.parser.add_argument("--wo_1st_update", action="store_true", help="without updating camera height after 1st epoch")
        self.parser.add_argument("--wo_1st2nd_update", action="store_true", help="without updating camera height after 2nd epoch")
        self.parser.add_argument(
            "--sparse_update", action="store_true", help="occasionally update camera height. When this argument is True, specify --update_freq."
        )
        self.parser.add_argument(
            "--update_freq", type=int, help="the frequency of updating camera height. This argument is valid only when --sparse_update is True"
        )
        self.parser.add_argument(
            "--damping_update",
            action="store_true",
            help="whether updating cam_height only when the updated value is smaller than twice thr original value",
        )
        self.parser.add_argument("--gamma", type=float, help="gamma for step_lr_scheduler", default=0.1)
        self.parser.add_argument("--start_with_cam_height_loss", action="store_true")  # TODO: this option is for debugging. remove this line.
        self.parser.add_argument("--warmup", action="store_true")
        self.parser.add_argument("--cam_height_loss_func", type=str, choices=["gaussian_nll_loss", "abs"], default="gaussian_nll_loss")
        self.parser.add_argument(
            "--rough_metric_loss_func", type=str, choices=["gaussian_nll_loss", "abs", "mean_after_abs"], default="gaussian_nll_loss"
        )
        self.parser.add_argument("--kernel_size", help="kernel size for erosion", type=int, default=5)
        self.parser.add_argument("--random_seed", help="random seed", type=int)
        self.parser.add_argument("--annot_height", action="store_true", help="whether using KITTI height labels")
        self.parser.add_argument(
            "--standard_metric",
            type=str,
            help="metric as an indicator of best model",
            choices=["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3", "loss"],
            default="loss",
        )

        self.parser.add_argument(
            "--segm_dirname",
            type=str,
            help="prefix of segm directory name like `{modified_segms_labels_person_car}",
            default="modified_segms_labels_person_car_road_th400",
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
        self.parser.add_argument("--rough_metric_scale_weight", type=float, help="weight of rough metric scale loss", default=1)
        self.parser.add_argument("--fine_metric_scale_weight", type=float, help="weight of fine metric scale loss", default=1)
        self.parser.add_argument(
            "--gradual_metric_scale_weight", action="store_true", help="whether increasing/decreasing fine/rough_metric_scale_weight gradually"
        )
        self.parser.add_argument("--gradual_weight_func", default="linear", choices=["linear", "sigmoid"])
        self.parser.add_argument(
            "--gradual_limit_epoch",
            type=int,
            help="upper limit of epoch for gradual increase when using --gradual_metric_scale_weight",
            default=20,
        )
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
        self.parser.add_argument("--num_epochs", type=int, help="number of epochs (including epochs of suspended training)", default=40)
        self.parser.add_argument("--scheduler_step_size", type=int, help="step size of the scheduler", default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale", help="if set, uses monodepth v1 multiscale", action="store_true")
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
        # self.parser.add_argument("--disp_filename_to_eval", type=str, help=".npy disparity filename to evaluate")
        self.parser.add_argument("--enable_loading_disp_to_eval", action="store_true")
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
