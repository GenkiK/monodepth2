import warnings

warnings.filterwarnings("ignore")

import json
import os
import pickle
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datasets
import networks
from layers import SSIM, BackprojectDepth, Project3D, compute_depth_errors, disp_to_depth, get_smooth_loss, transformation_from_parameters
from utils import (
    argmax_3d,
    calc_obj_pix_height_over_dist_to_horizon,
    cam_pts2cam_height_with_cross_prod,
    cam_pts2cam_heights,
    cam_pts2normal,
    generate_homo_pix_grid,
    masks_to_pix_heights,
    normalize_image,
    readlines,
    sec_to_hm_str,
    sigmoid,
    weighted_quantile,
)

segms_labels_str_set = ("segms", "labels")


def collate_fn(dict_batch):
    # batch: [{...} x batch_size]
    ret_dct = {key: default_collate([d[key] for d in dict_batch]) for key in dict_batch[0] if key not in segms_labels_str_set}
    padded_segms, padded_labels, n_insts = pad_segms_labels([d["segms"] for d in dict_batch], [d["labels"] for d in dict_batch])
    ret_dct["padded_segms"] = padded_segms
    ret_dct["padded_labels"] = padded_labels
    ret_dct["n_insts"] = n_insts
    return ret_dct


def pad_segms_labels(batch_segms, batch_labels):
    # assume that the no. insts is smaller than 255
    n_insts = torch.tensor([len(item) for item in batch_labels])
    padded_batch_segms = rnn.pad_sequence(batch_segms, batch_first=True, padding_value=0)
    padded_batch_labels = rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0)
    return padded_batch_segms, padded_batch_labels, n_insts


class TrainerHybrid:
    def __init__(self, options):
        self.opt = options
        if self.opt.sparse_update and self.opt.update_freq is None:
            raise ValueError("When --sparse_update is True, specify --update_freq.")
        if self.opt.damping_update:
            raise ValueError("This script doesn't support --damping_update.")
        if self.opt.wo_1st2nd_update:
            raise ValueError("This script doesn't support --wo_1st2nd_update.")

        if self.opt.soft_remove_outliers and self.opt.remove_outliers:
            raise ValueError("Choose either --soft_remove_outliers or --remove_outliers.")

        can_resume = self.opt.resume and self.opt.ckpt_timestamp

        if can_resume:
            self.log_path = os.path.join(
                self.opt.root_log_dir,
                f"{self.opt.width}x{self.opt.height}",
                f"{self.opt.model_name}{'_' if self.opt.model_name else ''}{self.opt.ckpt_timestamp}",
            )
            if not os.path.exists(self.log_path):
                raise FileNotFoundError(f"{self.log_path} does not exist.")
        else:
            self.log_path = os.path.join(
                options.root_log_dir,
                f"{options.width}x{options.height}",
                f"{options.model_name}{'_' if options.model_name else ''}{datetime.now().strftime('%m-%d-%H:%M')}",
            )

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.adj_frame_idxs)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.adj_frame_idxs[0] == 0, "adj_frame_idxs must start with 0"

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.opt.pose_model_type == "separate_resnet":
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume, num_input_images=self.num_pose_frames
            )

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

        elif self.opt.pose_model_type == "shared":
            self.models["pose"] = networks.PoseDecoder(self.models["encoder"].num_ch_enc, self.num_pose_frames)

        elif self.opt.pose_model_type == "posecnn":
            self.models["pose"] = networks.PoseCNN(self.num_input_frames if self.opt.pose_model_input == "all" else 2)

        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.epoch = 0

        if self.opt.init_after_1st_epoch:
            if self.opt.log_dirname_1st_epoch is None:
                # TODO: 学習済みのものがなかった場合に修正
                raise TypeError("self.log_dirname_1st_epoch is None. Specify log_dirname_1st_epoch option when --init_after_1st_epoch.")
            weights_dir_1st_epoch = (
                Path(self.opt.root_log_dir) / f"{self.opt.width}x{self.opt.height}" / f"{self.opt.log_dirname_1st_epoch}" / "models/weights_0"
            )
            self.load_cam_heights(weights_dir_1st_epoch, alert_if_not_exist=True)
            print("\nInitialized after 1st epoch!")
            print(f"Initial cam_height is {self.prev_mean_cam_height_expects_dict[0]}\n")

        if can_resume:
            if self.opt.last_epoch_for_resume is None:
                weights_dir, self.epoch = self.search_last_epoch()
            else:
                weights_dir = Path(self.log_path) / "models" / f"weights_{self.opt.last_epoch_for_resume}"
                self.epoch = self.opt.last_epoch_for_resume
            self.epoch += 1
            self.load_model(weights_dir)
            self.load_cam_heights(weights_dir, False)
        if self.opt.warmup and self.epoch <= 1:
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size - 2, self.opt.gamma)
        else:
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, self.opt.gamma)
        # self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.model_lr_scheduler.last_epoch = self.epoch - 1

        print("Training model named:  ", self.opt.model_name)
        if self.opt.dry_run:
            print("\n=====================================================================================\n")
            print("          This is dry-run mode, so no data will be saved               ")
            print("\n=====================================================================================\n")
        else:
            print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:  ", self.device)

        # data
        self.dataset = datasets.KITTIRAWDatasetWithRoad

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        if self.opt.dry_run:
            n_img = self.opt.batch_size * 2
            train_filenames = train_filenames[:n_img]
            val_filenames = val_filenames[:n_img]
        img_ext = ".png" if self.opt.png else ".jpg"

        num_train_samples = len(train_filenames)
        self.n_iter = num_train_samples // self.opt.batch_size
        self.num_total_steps = self.n_iter * self.opt.num_epochs
        self.min_road_area = int(self.opt.width * self.opt.height * 0.01)

        train_dataset = self.dataset(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.adj_frame_idxs,
            self.num_scales,
            is_train=True,
            img_ext=img_ext,
            segm_dirname=self.opt.segm_dirname,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.adj_frame_idxs,
            self.num_scales,
            is_train=False,
            img_ext=img_ext,
            segm_dirname=self.opt.segm_dirname,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        if not self.opt.dry_run:
            self.writers = {}
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2**scale)
            w = self.opt.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abse", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.standard_metric = self.opt.standard_metric
        if not (self.standard_metric in self.depth_metric_names or self.standard_metric == "loss"):
            raise KeyError(f"{self.standard_metric} is not in {self.depth_metric_names + ['loss']}")

        print("Using split:  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

        self.scale_init_dict = {scale: 0.0 for scale in self.opt.scales}
        self.sum_cam_height_expects_dict = self.scale_init_dict.copy()
        self.sum_cam_height_vars_dict = self.scale_init_dict.copy()

        if self.opt.annot_height:
            self.height_priors = torch.tensor(
                [
                    [1.747149944305419922e00, 6.863836944103240967e-02],
                    [np.nan, np.nan],
                    [1.5260834, 0.01868551],
                ],
                dtype=torch.float32,
            )
            print("Using annotated object height.\n")
        else:
            self.height_priors = TrainerHybrid.read_height_priors(self.opt.data_path)
            print("\nUsing calculated object height.\n")
        self.height_priors = self.height_priors.to(self.device)
        self.homo_pix_grid = generate_homo_pix_grid(self.opt.height, self.opt.width)

        self.outlier_th = self.opt.outlier_relative_error_th
        if not self.opt.dry_run:
            self.save_opts()

    @staticmethod
    def read_height_priors(root_dir: str) -> torch.Tensor:
        path = os.path.join(root_dir, "height_priors.txt")
        return torch.from_numpy(np.loadtxt(path, delimiter=" ", dtype=np.float32, ndmin=2))

    def set_train(self):
        """Convert all models to training mode"""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline"""
        self.step = self.epoch * len(self.train_loader)
        self.start_time = time.time()
        lower_is_better = self.standard_metric[:2] == "de" or self.standard_metric == "loss"
        th_best = 1000000 if lower_is_better else -1

        print(f"========= Training has started from {self.epoch} epoch. ========= ")
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.n_inst_frames = 0
            self.sum_cam_height_expects_dict = self.scale_init_dict.copy()
            self.sum_cam_height_vars_dict = self.scale_init_dict.copy()
            self.train_epoch()
            val_loss_dict = self.val_epoch()
            val_loss = val_loss_dict[self.standard_metric]
            new_mean_cam_height_expects_dict = {
                scale: sum_cam_height / self.n_inst_frames for scale, sum_cam_height in self.sum_cam_height_expects_dict.items()
            }
            new_mean_cam_height_vars_dict = {
                scale: sum_cam_height_var / self.n_inst_frames**2 for scale, sum_cam_height_var in self.sum_cam_height_vars_dict.items()
            }
            cond1 = (self.opt.wo_1st_update and self.epoch > 0) or (not self.opt.wo_1st_update)
            cond2 = (
                self.opt.sparse_update and (self.epoch - 1 if self.opt.wo_1st_update else self.epoch) % self.opt.update_freq == 0
            ) or not self.opt.sparse_update
            # if (cond1 or (cond2 and cond3)) and cond3:
            if cond1 and cond2:
                self.prev_mean_cam_height_expects_dict = new_mean_cam_height_expects_dict
                self.prev_mean_cam_height_vars_dict = new_mean_cam_height_vars_dict
                if not self.opt.dry_run:
                    self.log_cam_height()
            if not self.opt.dry_run:
                if (lower_is_better and val_loss < th_best) or (not lower_is_better and val_loss > th_best):
                    self.save_model(is_best=True)
                    th_best = val_loss
                else:
                    self.save_model(is_best=False)

    def train_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        self.set_train()

        for iteration, batch_input_dict in tqdm(enumerate(self.train_loader), dynamic_ncols=True):
            before_op_time = time.time()

            batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
            # for k in batch_input_dict:
            #     if k == "color" or k == "color_aug":
            #         batch_input_dict[k] = batch_input_dict[k].to(self.device, memory_format=torch.channels_last)
            #     else:
            #         batch_input_dict[k] = batch_input_dict[k].to(self.device)
            batch_output_dict, loss_dict = self.process_batch(batch_input_dict)

            self.model_optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = iteration % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step >= 2000 and self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(iteration, duration, loss_dict["loss"].cpu().data)

                if "depth_gt" in batch_input_dict:
                    self.compute_depth_losses(batch_input_dict, batch_output_dict, loss_dict)
                if not self.opt.dry_run:
                    self.log_train(batch_input_dict, batch_output_dict, loss_dict)
            del loss_dict
            torch.cuda.empty_cache()
            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, input_dict, mode="train"):
        """Pass a minibatch through the network and generate images and losses"""

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([input_dict[("color_aug", i, 0)] for i in self.opt.adj_frame_idxs])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {key: [f[i] for f in all_features] for i, key in enumerate(self.opt.adj_frame_idxs)}
            output_dict = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with adj_frame_idx 0 through the depth encoder
            features = self.models["encoder"](input_dict["color_aug", 0, 0])
            output_dict = self.models["depth"](features)

        output_dict.update(self.predict_poses(input_dict, features))

        self.generate_images_pred(input_dict, output_dict)
        loss_dict = self.compute_losses(input_dict, output_dict, mode)

        return output_dict, loss_dict

    def predict_poses(self, input_dict, features):
        """Predict poses between input frames for monocular sequences."""
        output_dict = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.adj_frame_idxs}
            else:
                pose_feats = {f_i: input_dict["color_aug", f_i, 0] for f_i in self.opt.adj_frame_idxs}

            for f_i in self.opt.adj_frame_idxs[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    output_dict[("axisangle", 0, f_i)] = axisangle
                    output_dict[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    output_dict[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat([input_dict[("color_aug", i, 0)] for i in self.opt.adj_frame_idxs if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.adj_frame_idxs if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.adj_frame_idxs[1:]):
                if f_i != "s":
                    output_dict[("axisangle", 0, f_i)] = axisangle
                    output_dict[("translation", 0, f_i)] = translation
                    output_dict[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])
        return output_dict

    def val_epoch(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        avg_loss_dict = {}
        with torch.no_grad():
            for batch_input_dict in self.val_loader:
                batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
                # for k in batch_input_dict:
                #     if k == "color" or k == "color_aug":
                #         batch_input_dict[k] = batch_input_dict[k].to(self.device, memory_format=torch.channels_last)
                #     else:
                #         batch_input_dict[k] = batch_input_dict[k].to(self.device)
                batch_output_dict, loss_dict = self.process_batch(batch_input_dict, mode="val")
                if "depth_gt" in batch_input_dict:
                    self.compute_depth_losses(batch_input_dict, batch_output_dict, loss_dict)
                for loss_name in loss_dict:
                    if loss_name in avg_loss_dict:
                        avg_loss_dict[loss_name] += loss_dict[loss_name]
                    else:
                        avg_loss_dict[loss_name] = loss_dict[loss_name]
                del batch_input_dict, batch_output_dict, loss_dict
            n_iter = len(self.val_loader)
            for loss_name in avg_loss_dict:
                avg_loss_dict[loss_name] /= n_iter
            if not self.opt.dry_run:
                self.log_val(avg_loss_dict)
            return avg_loss_dict

    def generate_images_pred(self, input_dict, output_dict):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        source_scale = 0
        for scale in self.opt.scales:
            disp = output_dict[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            cam_points = self.backproject_depth[source_scale](depth, input_dict[("inv_K", source_scale)])
            output_dict[("depth", scale)] = depth

            h = self.opt.height
            w = self.opt.width
            output_dict[("cam_pts", scale)] = cam_points[:, :-1, :].view(-1, 3, h, w).permute(0, 2, 3, 1)  # [bs, h, w, 3]
            for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                if adj_frame_idx == "s":
                    T = input_dict["stereo_T"]
                else:
                    T = output_dict[("cam_T_cam", 0, adj_frame_idx)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = output_dict[("axisangle", 0, adj_frame_idx)]
                    translation = output_dict[("translation", 0, adj_frame_idx)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], adj_frame_idx < 0)

                pix_coords = self.project_3d[source_scale](cam_points, input_dict[("K", source_scale)], T)

                output_dict[("sample", adj_frame_idx, scale)] = pix_coords

                output_dict[("color", adj_frame_idx, scale)] = F.grid_sample(
                    input_dict[("color", adj_frame_idx, source_scale)],
                    output_dict[("sample", adj_frame_idx, scale)],
                    padding_mode="border",
                    align_corners=True,
                )
                # if not self.opt.disable_automasking:
                #     output_dict[("color_identity", adj_frame_idx, scale)] = input_dict[("color", adj_frame_idx, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, input_dict, output_dict, mode):
        """Compute the reprojection and smoothness losses for a minibatch"""
        source_scale = 0
        loss_dict = {}
        total_loss = 0

        if self.opt.gradual_metric_scale_weight:
            match self.opt.gradual_weight_func:
                case "linear":
                    fine_metric_rate = min(self.epoch / self.opt.gradual_limit_epoch, 1)
                    rough_metric_rate = max(1 - self.epoch / self.opt.gradual_limit_epoch, 0)
                case "sigmoid":
                    fine_metric_rate = sigmoid(self.epoch - 3)
                    rough_metric_rate = 1 - sigmoid(self.epoch - 3)
        else:
            fine_metric_rate = 1
            rough_metric_rate = 1

        batch_road: torch.Tensor = input_dict["road"]
        batch_road_appear_bools = batch_road.sum((1, 2)) > self.min_road_area
        batch_road_wo_no_road = batch_road[batch_road_appear_bools]
        inv_Ks = input_dict[("inv_K", source_scale)][:, :3, :3][batch_road_appear_bools]  # [bs, 3, 3]

        h = self.opt.height
        w = self.opt.width
        img_area = h * w
        batch_n_insts: torch.Tensor = input_dict["n_insts"]
        n_inst_appear_frames = (batch_n_insts > 0).sum()
        if n_inst_appear_frames > 0:
            batch_segms: torch.Tensor = input_dict["padded_segms"]
            batch_labels: torch.Tensor = input_dict["padded_labels"]
            segms_flat, obj_pix_heights, obj_height_expects, obj_height_vars = self.make_flats(batch_segms, batch_labels)

            if self.opt.remove_outliers and batch_road_wo_no_road.shape[0] > 0 and hasattr(self, "prev_mean_cam_height_expects_dict"):
                bs_wo_no_road = batch_road_wo_no_road.shape[0]
                batch_cam_pts_wo_no_road: torch.Tensor = output_dict[("cam_pts", source_scale)].detach()[batch_road_appear_bools]
                normals = torch.zeros((bs_wo_no_road, 3, 1), device=self.device)
                for batch_wo_no_road_idx in range(bs_wo_no_road):
                    normals[batch_wo_no_road_idx] = cam_pts2normal(
                        batch_cam_pts_wo_no_road[batch_wo_no_road_idx], batch_road_wo_no_road[batch_wo_no_road_idx]
                    )
                cam_height = self.prev_mean_cam_height_expects_dict[0]
                horizons = (inv_Ks.transpose(1, 2) @ normals).squeeze()  # [bs_wo_no_road, 3]
                obj_pix_height_over_dist_to_horizon = calc_obj_pix_height_over_dist_to_horizon(
                    self.homo_pix_grid, batch_segms, horizons, batch_n_insts, batch_road_appear_bools
                )
                approx_heights = obj_pix_height_over_dist_to_horizon * cam_height
                relative_err = (approx_heights - obj_height_expects).abs() / obj_height_expects
                inlier_bools = relative_err < self.opt.outlier_relative_error_th
                segms_flat = segms_flat[inlier_bools]
                obj_pix_heights = obj_pix_heights[inlier_bools]
                obj_height_expects = obj_height_expects[inlier_bools]
                obj_height_vars = obj_height_vars[inlier_bools]
                split_inlier_mask = torch.split(inlier_bools, batch_n_insts.tolist())
                batch_n_insts = torch.tensor([chunk.sum() for chunk in split_inlier_mask], device=self.device)

            if self.opt.soft_remove_outliers and batch_road_wo_no_road.shape[0] > 0 and hasattr(self, "prev_mean_cam_height_expects_dict"):
                bs_wo_no_road = batch_road_wo_no_road.shape[0]
                batch_cam_pts_wo_no_road: torch.Tensor = output_dict[("cam_pts", source_scale)].detach()[batch_road_appear_bools]
                normals = torch.zeros((bs_wo_no_road, 3, 1), device=self.device)
                for batch_wo_no_road_idx in range(bs_wo_no_road):
                    normals[batch_wo_no_road_idx] = cam_pts2normal(
                        batch_cam_pts_wo_no_road[batch_wo_no_road_idx], batch_road_wo_no_road[batch_wo_no_road_idx]
                    )
                cam_height = self.prev_mean_cam_height_expects_dict[0]
                horizons = (inv_Ks.transpose(1, 2) @ normals).squeeze()  # [bs_wo_no_road, 3]
                obj_pix_height_over_dist_to_horizon = calc_obj_pix_height_over_dist_to_horizon(
                    self.homo_pix_grid, batch_segms, horizons, batch_n_insts, batch_road_appear_bools
                )
                approx_heights = obj_pix_height_over_dist_to_horizon * cam_height
                confidence_flat = 1 - ((approx_heights - obj_height_expects).abs() / obj_height_expects).clamp(max=self.outlier_th) / self.outlier_th
                # finally, soft_remove also removes outliers

            else:
                confidence_flat = torch.ones((segms_flat.shape[0],), device=self.device)

            # if self.opt.enable_erosion:
            #     batch_eroded_segms = erode(batch_segms[batch_road_appear_bools], self.opt.kernel_size)
            #     batch_eroded_n_insts = (batch_eroded_segms[batch_road_appear_bools].sum((2, 3)) > 0).sum(1)
            #     eroded_segms_flat = batch_eroded_segms.view(-1, h, w)
            #     vanished_channels = eroded_segms_flat.sum((1, 2)) > 0
            #     eroded_segms_flat = eroded_segms_flat[vanished_channels]
            #     eroded_obj_height_expects = obj_height_expects[batch_road_appear_bools][vanished_channels]
            #     eroded_obj_height_vars = obj_height_vars[batch_road_appear_bools][vanished_channels]
            #     eroded_obj_pix_heights = obj_pix_heights[batch_road_appear_bools][vanished_channels]

        if mode == "train":
            # TODO: erodeした場合，batch_n_instsが変わるのでこれを修正する必要がある
            self.n_inst_frames += (batch_n_insts[batch_road_appear_bools] > 0).sum()

        fy: float = input_dict[("K", source_scale)][0, 1, 1]
        batch_target: torch.Tensor = input_dict[("color", 0, source_scale)]
        for scale in self.opt.scales:
            loss = 0.0
            reprojection_losses = []
            batch_disp: torch.Tensor = output_dict[("disp", scale)]
            batch_upscaled_depth: torch.Tensor = output_dict[("depth", scale)].squeeze(1)
            depth_repeat = batch_upscaled_depth.repeat_interleave(batch_n_insts, dim=0)  # [sum(batch_n_insts), h, w]
            batch_color: torch.Tensor = input_dict[("color", 0, scale)]
            batch_cam_pts_wo_no_road: torch.Tensor = output_dict[("cam_pts", scale)][batch_road_appear_bools]  # [bs, h, w, 3]

            for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                batch_pred = output_dict[("color", adj_frame_idx, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

            reprojection_losses = torch.cat(reprojection_losses, 1)  # [bs, 2, h, w]

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                    batch_pred = input_dict[("color", adj_frame_idx, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # [bs, 2, h, w]

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001
                # combined_loss = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)  # [bs, 4, h, w]
                combined_loss = torch.cat((reprojection_loss, identity_reprojection_loss), dim=1)  # [bs, 4, h, w]
            else:
                combined_loss = reprojection_loss

            if combined_loss.shape[1] == 1:
                to_optimize = combined_loss
            else:
                to_optimize, idxs = torch.min(combined_loss, dim=1)  # idxs is already detached
                if self.opt.disable_road_masking:
                    # detachして(to_optimize), idxsを計算．このidxsをいじるようにする．道路領域についてはtorch.min(combined, dim=1)をtorch.min(reprojection_loss, dim=1)のidxsで代用する．
                    # detachしたcombinedで75%以上がマスクかかっているか計算する
                    automasks = (idxs < 2).float().sum((1, 2))  # [bs,]
                    batch_moving_bools = (1 - automasks / img_area) < 0.75
                    batch_road_moving_scene = (batch_moving_bools[:, None, None] * batch_road) > 0  # convert uint8 into bool for deprecation
                    to_optimize[batch_road_moving_scene] = reprojection_loss.min(1)[0][batch_road_moving_scene]

            # if not self.opt.disable_automasking:
            #     output_dict["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

            final_reprojection_loss = to_optimize.mean()
            loss_dict[f"loss/reprojection_{scale}"] = final_reprojection_loss.item()
            loss += final_reprojection_loss

            mean_disp = batch_disp.mean(2, True).mean(3, True)
            norm_disp = batch_disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, batch_color)

            loss_dict[f"loss/smoothness_{scale}"] = smooth_loss.item()
            loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)

            if batch_road_wo_no_road.shape[0] > 0:
                fine_metric_loss, normal_loss, unscaled_cam_heights = self.compute_cam_heights(
                    batch_cam_pts_wo_no_road,
                    batch_road_wo_no_road,
                )
                if self.opt.with_normal_loss:
                    loss_dict[f"loss/normal_{scale}"] = normal_loss.item()
                    if self.opt.gradual_normal_weight:
                        loss += fine_metric_rate * normal_loss
                    else:
                        loss += normal_loss
                if hasattr(self, "prev_mean_cam_height_expects_dict"):
                    loss_dict[f"loss/fine_metric_{scale}"] = fine_metric_loss.item()
                    loss += self.opt.fine_metric_scale_weight * fine_metric_rate * fine_metric_loss
            if n_inst_appear_frames == 0:
                loss_dict[f"loss/rough_metric_{scale}"] = 0.0
            else:
                rough_metric_loss = self.compute_rough_metric_loss(
                    depth_repeat,
                    segms_flat,
                    obj_pix_heights,
                    obj_height_expects,
                    obj_height_vars,
                    fy,
                    n_inst_appear_frames,
                    confidence_flat,
                )
                loss_dict[f"loss/rough_metric_{scale}"] = rough_metric_loss.item()
                loss += self.opt.rough_metric_scale_weight * rough_metric_rate * rough_metric_loss

                if self.opt.enable_erosion:
                    pass
                    # flat_road_appear_idxs = batch_road_appear_bools.repeat_interleave(batch_eroded_n_insts, dim=0)
                    # scaled_sum_cam_height_expects, scaled_sum_cam_height_vars = self.scale_cam_heights(
                    #     batch_cam_pts_wo_no_road,
                    #     depth_repeat.detach()[flat_road_appear_idxs],
                    #     eroded_segms_flat,
                    #     eroded_obj_pix_heights,
                    #     eroded_obj_height_expects,
                    #     eroded_obj_height_vars,
                    #     batch_eroded_n_insts,
                    #     unscaled_cam_heights,
                    #     fy,
                    #     confidence_flat,
                    # )
                else:
                    flat_road_appear_idxs = batch_road_appear_bools.repeat_interleave(batch_n_insts, dim=0)
                    scaled_sum_cam_height_expects, scaled_sum_cam_height_vars = self.scale_cam_heights(
                        batch_cam_pts_wo_no_road,
                        depth_repeat.detach()[flat_road_appear_idxs],
                        segms_flat[flat_road_appear_idxs],
                        obj_pix_heights[flat_road_appear_idxs],
                        obj_height_expects[flat_road_appear_idxs],
                        obj_height_vars[flat_road_appear_idxs],
                        batch_n_insts[batch_road_appear_bools],
                        unscaled_cam_heights,
                        fy,
                        confidence_flat[flat_road_appear_idxs],
                    )

                if mode == "train" and scaled_sum_cam_height_expects is not None:
                    self.sum_cam_height_expects_dict[scale] += scaled_sum_cam_height_expects
                    self.sum_cam_height_vars_dict[scale] += scaled_sum_cam_height_vars

            total_loss += loss
            loss_dict[f"loss/{scale}"] = loss.item()

        total_loss /= self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict

    def compute_cam_heights(
        self,
        batch_cam_pts: torch.Tensor,  # [bs, h, w, 3]
        batch_road: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = batch_cam_pts.shape[0]
        frame_unscaled_cam_heights = torch.zeros(bs, device=self.device)
        cam_height_loss = torch.tensor(0.0, device=self.device)
        normal_loss = torch.tensor(0.0, device=self.device)
        if self.opt.normal_with_8_neighbors:
            batch_cam_height, batch_normal = cam_pts2cam_height_with_cross_prod(batch_cam_pts)
            batch_road[:, -1, :] = 0
            batch_road[:, :, 0] = 0
            batch_road[:, :, -1] = 0
        for batch_idx in range(bs):
            if self.opt.normal_with_8_neighbors:
                road_cam_heights = batch_cam_height[batch_idx, batch_road[batch_idx] == 1]  # [?,]
                normals = batch_normal[batch_idx, batch_road[batch_idx] == 1]  # [?, 3]
                normal_loss = normal_loss + torch.abs(normals - normals.detach().quantile(0.5, dim=0)).mean(0).mean()
            else:
                road_cam_heights = cam_pts2cam_heights(batch_cam_pts[batch_idx], batch_road[batch_idx])  # [?, 3]
            if hasattr(self, "prev_mean_cam_height_expects_dict"):
                match self.opt.cam_height_loss_func:
                    case "gaussian_nll_loss":
                        cam_height_loss = cam_height_loss + F.gaussian_nll_loss(
                            input=self.prev_mean_cam_height_expects_dict[0],
                            target=road_cam_heights,
                            var=self.prev_mean_cam_height_vars_dict[0],
                            eps=0.001,
                            reduction="mean",
                        )
                    case "abs":
                        cam_height_loss = cam_height_loss + torch.abs(self.prev_mean_cam_height_expects_dict[0] - road_cam_heights).mean()
            if self.opt.use_median_cam_height:
                frame_unscaled_cam_heights[batch_idx] = road_cam_heights.detach().quantile(0.5)
            else:
                frame_unscaled_cam_heights[batch_idx] = road_cam_heights.detach().mean()
        return cam_height_loss / bs, normal_loss / bs, frame_unscaled_cam_heights

    def scale_cam_heights(
        self,
        batch_cam_pts: torch.Tensor,  # [bs, h, w, 3]
        depth_repeat_detach: torch.Tensor,
        segms_flat: torch.Tensor,
        obj_pix_heights: torch.Tensor,
        obj_height_expects: torch.Tensor,
        obj_height_vars: torch.Tensor,
        batch_n_insts: torch.Tensor,
        frame_unscaled_cam_heights: torch.Tensor,
        fy: float,
        confidence_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if segms_flat.shape[0] == 0:
            return None, None
        depth_expects = obj_height_expects * fy / obj_pix_heights

        if self.opt.use_median_depth or self.opt.use_1st_quartile_depth:
            q = 0.5 if self.opt.use_median_depth else 0.25
            masked_depth_repeat = depth_repeat_detach * segms_flat
            nearest_depths = torch.zeros((depth_repeat_detach.shape[0],), dtype=torch.float32, device=self.device)
            for i in range(masked_depth_repeat.shape[0]):
                nearest_depths[i] = masked_depth_repeat[i][masked_depth_repeat[i] > 0].quantile(q=q)
        else:
            cam_pts_repeat = batch_cam_pts.detach().repeat_interleave(batch_n_insts, dim=0)
            masked_cam_pts = cam_pts_repeat * segms_flat.unsqueeze(-1)
            masked_cam_pts[masked_cam_pts == 0] = 1000
            nearest_pts = argmax_3d(-torch.linalg.norm(masked_cam_pts, dim=3))  # [n_insts * bs, 2]
            nearest_depths = depth_repeat_detach[
                torch.arange(depth_repeat_detach.shape[0]), nearest_pts[:, 0], nearest_pts[:, 1]
            ]  # [n_insts * bs, 2]

        batch_n_insts_lst = batch_n_insts.tolist()
        split_scale_expects = torch.split(depth_expects / nearest_depths, batch_n_insts_lst)
        split_confidence = torch.split(confidence_flat, batch_n_insts_lst)
        if self.opt.use_median_scale:
            # frame_scale_expects = torch.tensor([chunk.median() for chunk in split_scale_expects], device=self.device)
            frame_scale_expects = torch.tensor(
                [weighted_quantile(chunk, w) for chunk, w in zip(split_scale_expects, split_confidence)], device=self.device
            )
        else:
            # frame_scale_expects = torch.tensor([chunk.mean() for chunk in split_scale_expects], device=self.device)
            frame_scale_expects = torch.tensor([(chunk @ w) / w.sum() for chunk, w in zip(split_scale_expects, split_confidence)], device=self.device)
        split_scale_vars = torch.split(obj_height_vars / (obj_pix_heights * nearest_depths / fy) ** 2, batch_n_insts_lst)
        # frame_scale_var = torch.tensor([chunk.sum() for chunk in split_scale_vars], device=obj_height_vars.device) / batch_n_insts**2
        frame_scale_var = (
            torch.tensor([(chunk @ w) / w.sum() for chunk, w in zip(split_scale_vars, split_confidence)], device=self.device) / batch_n_insts**2
        )

        scaled_sum_cam_height_expects = (frame_scale_expects * frame_unscaled_cam_heights).nansum()
        scaled_sum_cam_height_vars = (frame_scale_var * frame_unscaled_cam_heights**2).nansum()
        return scaled_sum_cam_height_expects, scaled_sum_cam_height_vars

    def compute_rough_metric_loss(
        self,
        depth_repeat: torch.Tensor,
        segms_flat: torch.Tensor,
        obj_pix_heights: torch.Tensor,
        obj_height_expects: torch.Tensor,
        obj_height_vars: torch.Tensor,
        fy: float,
        n_inst_appear_frames: int,
        confidence_flat: torch.Tensor,
    ) -> torch.Tensor:
        match self.opt.rough_metric_loss_func:
            case "gaussian_nll_loss":
                obj_mean_depths = (depth_repeat * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2)).clamp(min=1e-9)
                pred_heights = obj_pix_heights * obj_mean_depths / fy
                loss = F.gaussian_nll_loss(input=obj_height_expects, target=pred_heights, var=obj_height_vars, reduction="none") * confidence_flat
            case "abs":
                obj_mean_depths = (depth_repeat * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2)).clamp(min=1e-9)
                pred_heights = obj_pix_heights * obj_mean_depths / fy
                loss = torch.abs(obj_height_expects - pred_heights) * confidence_flat
            case "mean_after_abs":
                pred_heights = obj_pix_heights[:, None, None] * depth_repeat / fy  # [sum(batch_n_insts), h, w]
                loss = (
                    torch.abs((obj_height_expects[:, None, None] - pred_heights) * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2))
                ) * confidence_flat
        assert not loss.mean().isnan()
        return loss.mean() / n_inst_appear_frames

    def make_flats(
        self,
        batch_segms: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h, w = batch_segms.shape
        segms_flat = batch_segms.view(-1, h, w)
        non_padded_channels = segms_flat.sum(dim=(1, 2)) > 0
        segms_flat = segms_flat[non_padded_channels]  # exclude padded channels
        labels_flat = batch_labels.view(-1)[non_padded_channels].long()

        obj_pix_heights = masks_to_pix_heights(segms_flat)
        obj_height_expects = self.height_priors[labels_flat, 0]
        obj_height_vars = self.height_priors[labels_flat, 1]
        return segms_flat, obj_pix_heights, obj_height_expects, obj_height_vars

    def compute_depth_losses(self, input_dict, output_dict, loss_dict):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        batch_depth_pred = output_dict[("depth", 0)].detach()
        batch_depth_pred = torch.clamp(F.interpolate(batch_depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)

        batch_depth_gt = input_dict["depth_gt"]
        batch_mask = batch_depth_gt > 0

        # garg/eigen crop
        batch_crop_mask = torch.zeros_like(batch_mask)
        batch_crop_mask[:, :, 153:371, 44:1197] = 1
        batch_mask = batch_mask * batch_crop_mask

        batch_depth_gt = batch_depth_gt[batch_mask]
        batch_depth_pred = batch_depth_pred[batch_mask]
        # turnoff median scaling
        # batch_depth_pred *= torch.median(batch_depth_gt) / torch.median(batch_depth_pred)

        batch_depth_pred = torch.clamp(batch_depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(batch_depth_gt, batch_depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            loss_dict[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(
            print_string.format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
                sec_to_hm_str(time_sofar),
                sec_to_hm_str(training_time_left),
            )
        )

    def log_cam_height(self):
        for writer in self.writers.values():
            for scale in self.opt.scales:
                writer.add_scalar(f"cam_height_expect_{scale}", self.prev_mean_cam_height_expects_dict[scale], self.step)
                writer.add_scalar(f"cam_height_var_{scale}", self.prev_mean_cam_height_vars_dict[scale], self.step)
            writer.add_scalar("n_inst_frames", self.n_inst_frames, self.step)

    def log_train(self, input_dict, output_dict, loss_dict):
        """Write an event to the tensorboard events file"""
        writer = self.writers["train"]
        for name, loss in loss_dict.items():
            writer.add_scalar(name, loss, self.step)

        if self.opt.log_image:
            for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
                for scale in self.opt.scales:
                    for frame_id in self.opt.adj_frame_idxs:
                        writer.add_image(f"color_{frame_id}_{scale}/{j}", input_dict[("color", frame_id, scale)][j].data, self.step)
                        if scale == 0 and frame_id != 0:
                            writer.add_image(
                                f"color_pred_{frame_id}_{scale}/{j}",
                                output_dict[("color", frame_id, scale)][j].data,
                                self.step,
                            )

                    writer.add_image(f"disp_{scale}/{j}", normalize_image(output_dict[("disp", scale)][j]), self.step)
                    if not self.opt.disable_automasking:
                        writer.add_image(
                            f"automask_{scale}/{j}",
                            output_dict[f"identity_selection/{scale}"][j][None, ...],
                            self.step,
                        )

    def log_val(self, loss_dict):
        """Write an event to the tensorboard events file"""
        for name, loss in loss_dict.items():
            self.writers["val"].add_scalar(name, loss, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, "opt.json"), "w") as f:
            json.dump(to_save, f, indent=2)

    def load_opts(self):
        json_path = os.path.join(self.log_path, "models", "opt.json")
        try:
            with open(json_path, "r") as f:
                self.opt = Namespace()
                for k, v in json.load(f):
                    setattr(self.opt, k, v)
                print("TODO: namespaceがきちんと読み込まれているかを確認")
        except FileNotFoundError:
            print(f"FileNotFoundError: option json file path {self.log_path} does not exist.")

    def save_model(self, is_best=False):
        """Save model weights to disk"""
        if is_best:
            save_best_folder = os.path.join(self.log_path, "models", "best_weights")
            if not os.path.exists(save_best_folder):
                os.makedirs(save_best_folder)
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == "encoder":
                # save the sizes - these are needed at prediction time
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
            torch.save(to_save, save_path)
            if is_best:
                torch.save(to_save, os.path.join(save_best_folder, f"{model_name}.pth"))

        if hasattr(self, "prev_mean_cam_height_expects_dict"):
            cam_height_expect_path = os.path.join(save_folder, "cam_height_expect.pkl")
            with open(cam_height_expect_path, "wb") as f:
                pickle.dump(self.prev_mean_cam_height_expects_dict, f, pickle.HIGHEST_PROTOCOL)

            cam_height_var_path = os.path.join(save_folder, "cam_height_var.pkl")
            with open(cam_height_var_path, "wb") as f:
                pickle.dump(self.prev_mean_cam_height_vars_dict, f, pickle.HIGHEST_PROTOCOL)

        optimizer_save_path = os.path.join(save_folder, "adam.pth")
        torch.save(self.model_optimizer.state_dict(), optimizer_save_path)
        if is_best:
            torch.save(self.model_optimizer.state_dict(), os.path.join(save_best_folder, "adam.pth"))

    def search_last_epoch(self) -> tuple[Path, int]:
        root_weights_dir = Path(self.log_path) / "models"
        last_epoch = -1
        for weights_dir in root_weights_dir.glob("weights_*"):
            epoch = int(weights_dir.name[8:])
            if epoch > last_epoch:
                last_epoch = epoch
        return root_weights_dir / f"weights_{last_epoch}", last_epoch

    def load_cam_heights(self, weights_dir: Path, alert_if_not_exist: bool):
        # load prev_cam_height
        cam_height_expect_path = weights_dir / "cam_height_expect.pkl"
        if cam_height_expect_path.exists():
            print("Loading prev_mean_cam_height_expects_dict")
            with open(cam_height_expect_path, "rb") as f:
                self.prev_mean_cam_height_expects_dict = pickle.load(f)
        elif alert_if_not_exist:
            raise FileNotFoundError(f"\n{cam_height_expect_path} does not exists.\n")
        else:
            print(f"\n{cam_height_expect_path} does not exists.\n")
        cam_height_var_path = weights_dir / "cam_height_var.pkl"
        if cam_height_var_path.exists():
            print("Loading prev_mean_cam_height_vars_dict")
            with open(cam_height_var_path, "rb") as f:
                self.prev_mean_cam_height_vars_dict = pickle.load(f)
        elif alert_if_not_exist:
            raise FileNotFoundError(f"\n{cam_height_var_path} does not exists.\n")
        else:
            print(f"\n{cam_height_var_path} does not exists.\n")

    def load_model(self, weights_dir: Path):
        """Load model(s) from disk"""

        print(f"loading model from folder {weights_dir}")

        for model_name in self.opt.models_to_load:
            print(f"Loading {model_name} weights...")
            path = weights_dir / f"{model_name}.pth"
            model_dict = self.models[model_name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[model_name].load_state_dict(model_dict)

        # load adam state
        optimizer_load_path = weights_dir / "adam.pth"
        if optimizer_load_path.exists():
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
