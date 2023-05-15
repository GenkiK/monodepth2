# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import json
import os
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from utils import masks_to_pix_heights, normalize_image, readlines, sec_to_hm_str

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
    n_insts = torch.tensor([len(item) for item in batch_labels])
    padded_batch_segms = rnn.pad_sequence(batch_segms, batch_first=True, padding_value=0)
    padded_batch_labels = rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0)
    return padded_batch_segms, padded_batch_labels, n_insts


class TrainerWithSegm:
    def __init__(self, options):
        can_resume = options.resume and options.ckpt_timestamp
        self.opt = options

        if can_resume:
            # self.load_opts()
            self.log_path = os.path.join(
                self.opt.root_log_dir,
                f"{self.opt.width}x{self.opt.height}",
                f"{self.opt.model_name}{'_' if self.opt.model_name else ''}{self.opt.ckpt_timestamp}",
            )
            if not os.path.exists(self.log_path):
                raise FileNotFoundError(f"{self.log_path} does not exist.")
        else:
            # self.opt = options
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

        self.use_pose_net = not (self.opt.use_stereo and self.opt.adj_frame_idxs == [0])

        if self.opt.use_stereo:
            self.opt.adj_frame_idxs.append("s")

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained" and not can_resume)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
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

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.opt.scales,
                num_output_channels=(len(self.opt.adj_frame_idxs) - 1),
            )
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.epoch = 0
        if can_resume:
            weights_dir, self.epoch = self.search_last_epoch()
            self.epoch += 1
            self.load_model(weights_dir)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.model_lr_scheduler.last_epoch = self.epoch - 1

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDatasetWithSegm, "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png" if self.opt.png else ".jpg"

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

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
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,  # TODO: とりあえずvalでもsegmsを読み込むようにしている．いらないようであれば修正（compute_losses()でsegmsを呼び出している部分をなくす）
        )
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

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

        if self.opt.annot_height:
            self.height_priors = np.array(
                [
                    [1.747149944305419922e00, 6.863836944103240967e-02],
                    [np.nan, np.nan],
                    [1.5260834, 0.01868551],
                ]
            )
            print("\nUsing annotated object height.\n")
        else:
            self.height_priors = TrainerWithSegm.read_height_priors(self.opt.data_path)
            print("\nUsing calculated object height.\n")
        self.height_priors = self.height_priors.to(self.device)

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

        print(f"========= Training has started from {self.epoch}epoch. ========= ")
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            val_loss_dict = self.val()
            val_loss = val_loss_dict[self.standard_metric]
            if (lower_is_better and val_loss < th_best) or (not lower_is_better and val_loss > th_best):
                self.save_model(is_best=True)
                th_best = val_loss
            else:
                self.save_model(is_best=False)

    def run_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        self.set_train()

        for batch_idx, batch_input_dict in tqdm(enumerate(self.train_loader), dynamic_ncols=True):
            before_op_time = time.time()

            batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
            output_dict, loss_dict = self.process_batch(batch_input_dict)

            self.model_optimizer.zero_grad()
            loss_dict["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step >= 2000 and self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss_dict["loss"].cpu().data)

                if "depth_gt" in batch_input_dict:
                    self.compute_depth_losses(batch_input_dict, output_dict, loss_dict)
                self.log_train(batch_input_dict, output_dict, loss_dict)
            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, input_dict):
        """Pass a minibatch through the network and generate images and losses"""
        # input_dict = {key: ipt.to(self.device) for key, ipt in input_dict.items()}

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

        if self.opt.predictive_mask:
            output_dict["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            output_dict.update(self.predict_poses(input_dict, features))

        self.generate_images_pred(input_dict, output_dict)
        loss_dict = self.compute_losses(input_dict, output_dict)

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

    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        avg_loss_dict = {}
        with torch.no_grad():
            for batch_input_dict in self.val_loader:
                batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
                batch_output_dict, loss_dict = self.process_batch(batch_input_dict)
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
            self.log_val(avg_loss_dict)
            self.set_train()
            return avg_loss_dict

    def generate_images_pred(self, input_dict, output_dict):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = output_dict[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            output_dict[("depth", scale)] = depth

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

                cam_points = self.backproject_depth[source_scale](depth, input_dict[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, input_dict[("K", source_scale)], T)

                output_dict[("sample", adj_frame_idx, scale)] = pix_coords

                output_dict[("color", adj_frame_idx, scale)] = F.grid_sample(
                    input_dict[("color", adj_frame_idx, source_scale)],
                    output_dict[("sample", adj_frame_idx, scale)],
                    padding_mode="border",
                    align_corners=True,
                )

                if not self.opt.disable_automasking:
                    output_dict[("color_identity", adj_frame_idx, scale)] = input_dict[("color", adj_frame_idx, source_scale)]

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

    def compute_losses(self, input_dict, output_dict):
        """Compute the reprojection and smoothness losses for a minibatch"""
        loss_dict = {}
        total_loss = 0

        batch_segms: torch.Tensor = input_dict["padded_segms"]
        batch_labels: torch.Tensor = input_dict["padded_labels"]
        batch_n_insts: torch.Tensor = input_dict["n_insts"]

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            batch_disp: torch.Tensor = output_dict[("disp", scale)]
            batch_upscaled_depth: torch.Tensor = output_dict[("depth", scale)]
            batch_K: torch.Tensor = input_dict[("K", source_scale)]
            batch_color: torch.Tensor = input_dict[("color", 0, scale)]
            batch_target: torch.Tensor = input_dict[("color", 0, source_scale)]

            for frame_idx in self.opt.adj_frame_idxs[1:]:
                batch_pred = output_dict[("color", frame_idx, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_idx in self.opt.adj_frame_idxs[1:]:
                    batch_pred = input_dict[("color", frame_idx, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = output_dict["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimize = combined
            else:
                to_optimize, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                output_dict["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

            # loss += to_optimize.mean()
            final_reprojection_loss = to_optimize.mean()
            loss_dict[f"loss/reprojection_{scale}"] = final_reprojection_loss
            loss += final_reprojection_loss

            mean_disp = batch_disp.mean(2, True).mean(3, True)
            norm_disp = batch_disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, batch_color)

            loss_dict[f"loss/smoothness_{scale}"] = smooth_loss
            loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)

            rough_metric_loss = self.compute_rough_metric_loss(batch_upscaled_depth, batch_segms, batch_labels, batch_n_insts, batch_K)
            loss_dict[f"loss/rough_metric_{scale}"] = rough_metric_loss
            loss += self.opt.rough_metric_scale_weight * rough_metric_loss / (2**scale)

            total_loss += loss
            loss_dict[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict

    def compute_rough_metric_loss(
        self, batch_depth: torch.Tensor, batch_segms: torch.Tensor, batch_labels: torch.Tensor, batch_n_insts: torch.Tensor, batch_K: torch.Tensor
    ) -> float:
        bs, _, h, w = batch_depth.shape
        batch_depth = batch_depth.view(bs, h, w)

        assert 1 <= bs < h < w
        assert batch_n_insts.sum() < bs * batch_segms.shape[1]  # batch_segmsがcompressされていないか

        if batch_n_insts.sum() == 0:
            return 0.0
        fy_repeat = batch_K[:, 1, 1].repeat_interleave(batch_n_insts, dim=0)
        assert fy_repeat.shape == (batch_n_insts.sum(),)

        depth_repeat = batch_depth.repeat_interleave(batch_n_insts, dim=0)  # [sum(batch_n_insts), h, w]
        assert depth_repeat.shape == (batch_n_insts.sum(), h, w)

        segms_flat = batch_segms.view(-1, h, w)
        non_padded_channels = segms_flat.sum(dim=(1, 2)) > 0
        segms_flat = segms_flat[non_padded_channels]  # exclude padded channels
        assert segms_flat.shape == (batch_n_insts.sum(), h, w)

        obj_mean_depths = (depth_repeat * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2)).clamp(min=1e-9)

        obj_pix_heights = masks_to_pix_heights(segms_flat)

        assert (obj_pix_heights < 0).sum() == 0

        labels_flat = batch_labels.view(-1)[non_padded_channels].long()
        height_priors = self.height_priors[labels_flat, :]

        pred_heights = obj_pix_heights * obj_mean_depths / fy_repeat
        # loss = F.gaussian_nll_loss(input=height_priors[:, 0], target=pred_heights, var=height_priors[:, 1], eps=0.001) / (batch_n_insts > 0).sum()
        loss = F.gaussian_nll_loss(input=height_priors[:, 0], target=pred_heights, var=height_priors[:, 1], reduction="mean") / self.batch_size
        assert not loss.isnan()
        return loss

    def compute_depth_losses(self, input_dict, output_dict, loss_dict):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        batch_depth_pred = output_dict[("depth", 0)]
        batch_depth_pred = torch.clamp(F.interpolate(batch_depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        batch_depth_pred = batch_depth_pred.detach()

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

    def log_train(self, input_dict, output_dict, loss_dict):
        """Write an event to the tensorboard events file"""
        writer = self.writers["train"]
        for name, loss in loss_dict.items():
            writer.add_scalar(f"{name}", loss, self.step)

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

                    writer.add_image("disp_{}/{}".format(scale, j), normalize_image(output_dict[("disp", scale)][j]), self.step)

                    if self.opt.predictive_mask:
                        for f_idx, frame_id in enumerate(self.opt.adj_frame_idxs[1:]):
                            writer.add_image(
                                f"predictive_mask_{frame_id}_{scale}/{j}",
                                output_dict["predictive_mask"][("disp", scale)][j, f_idx][None, ...],
                                self.step,
                            )

                    elif not self.opt.disable_automasking:
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
                breakpoint()
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
                to_save["use_stereo"] = self.opt.use_stereo
            torch.save(to_save, save_path)
            if is_best:
                torch.save(to_save, os.path.join(save_best_folder, f"{model_name}.pth"))

        optimizer_save_path = os.path.join(save_folder, "adam.pth")
        torch.save(self.model_optimizer.state_dict(), optimizer_save_path)
        if is_best:
            torch.save(self.model_optimizer.state_dict(), os.path.join(save_best_folder, "adam.pth"))
        # scheduler_save_path = os.path.join(save_folder, "scheduler.pth")
        # torch.save(self.model_lr_scheduler.state_dict(), scheduler_save_path)

    # def load_model(self):
    #     """Load model(s) from disk"""
    #     self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

    #     assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
    #     print("loading model from folder {}".format(self.opt.load_weights_folder))

    #     for n in self.opt.models_to_load:
    #         print("Loading {} weights...".format(n))
    #         path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
    #         model_dict = self.models[n].state_dict()
    #         pretrained_dict = torch.load(path)
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #         model_dict.update(pretrained_dict)
    #         self.models[n].load_state_dict(model_dict)

    #     # loading adam state
    #     optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
    #     if os.path.isfile(optimizer_load_path):
    #         print("Loading Adam weights")
    #         optimizer_dict = torch.load(optimizer_load_path)
    #         self.model_optimizer.load_state_dict(optimizer_dict)
    #     else:
    #         print("Cannot find Adam weights so Adam is randomly initialized")

    def search_last_epoch(self) -> tuple[Path, int]:
        root_weights_dir = Path(self.log_path) / "models"
        last_epoch = -1
        for weights_dir in root_weights_dir.glob("weights_*"):
            epoch = int(weights_dir.name[8:])
            if epoch > last_epoch:
                last_epoch = epoch
        return root_weights_dir / f"weights_{last_epoch}", last_epoch

    def load_model(self, weights_dir: Path):
        """Load model(s) from disk"""

        print(f"loading model from folder {weights_dir}")

        for model_name in self.opt.models_to_load:
            print("Loading {} weights...".format(model_name))
            path = weights_dir / f"{model_name}.pth"
            model_dict = self.models[model_name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[model_name].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = weights_dir / "adam.pth"
        if optimizer_load_path.exists():
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

        # # loading scheduler state
        # scheduler_load_path = weights_dir.parent / "scheduler.pth"
        # if scheduler_load_path.exists():
        #     print("Loading learning rate scheduler weights")
        #     scheduler_dict = torch.load(scheduler_load_path)
        #     self.model_lr_scheduler.load_state_dict(scheduler_dict)
        # else:
        #     print(f"Cannot find scheduler weights so it is initialized with last_epoch={self.epoch - 1}")
