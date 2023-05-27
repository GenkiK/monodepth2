# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import datasets
import networks
from layers import SSIM, BackprojectDepth, Project3D, compute_depth_errors, disp_to_depth, get_smooth_loss, transformation_from_parameters
from utils import argmax_3d, cam_pts2cam_heights, depths2cam_pts, erode, generate_cam_grid, masks_to_pix_heights, readlines

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


class TrainerWithRoadDebug:
    def __init__(self, options):
        if options.local:
            options.data_path = options.data_path.replace("gkinoshita", "gkinoshita/humpback")
            options.root_log_dir = options.root_log_dir.replace("gkinoshita", "gkinoshita/humpback")

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
            if self.opt.last_epoch_for_resume is None:
                weights_dir, self.epoch = self.search_last_epoch()
            else:
                weights_dir = Path(self.log_path) / "models" / f"weights_{self.opt.last_epoch_for_resume}"
                self.epoch = self.opt.last_epoch_for_resume
            self.epoch += 1
            self.load_model(weights_dir)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.model_lr_scheduler.last_epoch = self.epoch - 1

        print("Training model named:\n  ", self.opt.model_name)
        print("Training is using:\n  ", self.device)

        # data
        self.dataset = datasets.KITTIRAWDatasetWithRoad

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
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

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

        self.scale_init_dict = {scale: 0.0 for scale in self.opt.scales}
        self.sum_cam_height_expects_dict = self.scale_init_dict.copy()
        self.sum_cam_height_vars_dict = self.scale_init_dict.copy()
        K = torch.tensor([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]], dtype=torch.float32)
        K[0, :] *= self.opt.width
        K[1, :] *= self.opt.height
        invK = torch.linalg.pinv(K)
        self.cam_grid = generate_cam_grid(self.opt.height, self.opt.width, invK).to(self.device)

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
            self.height_priors = TrainerWithRoadDebug.read_height_priors(self.opt.data_path)
            print("\nUsing calculated object height.\n")
        self.height_priors = self.height_priors.to(self.device)
        self.kernel_size = self.opt.kernel_size  # default: 5

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

        print(f"========= Training has started from {self.epoch} epoch. ========= ")
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.n_inst_frames = 0
            self.sum_cam_height_expects_dict = self.scale_init_dict.copy()
            self.sum_cam_height_vars_dict = self.scale_init_dict.copy()
            self.train_epoch()
            self.val_epoch()
            self.prev_mean_cam_height_expects_dict = {
                k: sum_cam_height / self.n_inst_frames for k, sum_cam_height in self.sum_cam_height_expects_dict.items()
            }
            self.prev_mean_cam_height_vars_dict = {
                k: sum_cam_height_var / self.n_inst_frames**2 for k, sum_cam_height_var in self.sum_cam_height_vars_dict.items()
            }
            print(f"{self.prev_mean_cam_height_expects_dict=}")
            print(f"{self.prev_mean_cam_height_vars_dict=}")

    def train_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        self.set_train()

        for batch_input_dict in tqdm(self.train_loader, dynamic_ncols=True):
            batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
            output_dict, loss_dict = self.process_batch(batch_input_dict)

            self.model_optimizer.zero_grad(set_to_none=True)
            # with torch.autograd.detect_anomaly():
            #     loss_dict["loss"].backward()
            loss_dict["loss"].backward()
            self.model_optimizer.step()
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

        if self.opt.predictive_mask:
            output_dict["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
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
            return avg_loss_dict

    def generate_images_pred(self, input_dict, output_dict):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = output_dict[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            output_dict[("depth", scale)] = depth

            cam_points = self.backproject_depth[source_scale](depth, input_dict[("inv_K", source_scale)])
            h = self.opt.height
            w = self.opt.width
            # TODO: [:, :-1, :]の意味
            # output_dict[("cam_pts", scale)] = cam_points[:, :-1, :].view(-1, 3, h, w)
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

    def compute_losses(self, input_dict, output_dict, mode):
        """Compute the reprojection and smoothness losses for a minibatch"""
        source_scale = 0
        loss_dict = {}
        total_loss = 0

        batch_segms: torch.Tensor = input_dict["padded_segms"]
        batch_road: torch.Tensor = input_dict["road"]
        batch_labels: torch.Tensor = input_dict["padded_labels"]
        batch_n_insts: torch.Tensor = input_dict["n_insts"]

        # bs = self.opt.batch_size
        # plt.figure(figsize=(15 * bs, 5 * 3))
        # for idx in range(bs):
        #     plt.subplot(3, bs, idx + 1)
        #     plt.imshow(batch_segms.detach().cpu().numpy()[idx].sum(0))
        #     plt.tight_layout()
        #     plt.axis("off")
        # # plt.show()
        # # plt.close()

        # # plt.figure(figsize=(15 * bs, 8))
        # for idx in range(bs, bs * 2):
        #     plt.subplot(3, bs, idx + 1)
        #     plt.imshow(batch_road.detach().cpu().numpy()[idx - bs])
        #     plt.tight_layout()
        #     plt.axis("off")

        # for idx in range(bs * 2, bs * 3):
        #     plt.subplot(3, bs, idx + 1)
        #     plt.imshow(output_dict[("disp", 0)].squeeze().detach().cpu().numpy()[idx - bs * 2], cmap="Spectral")
        #     plt.tight_layout()
        #     plt.axis("off")
        # plt.show()
        # plt.close()

        if mode == "train":
            self.n_inst_frames += (batch_n_insts > 0).sum()

        fy: float = input_dict[("K", source_scale)][0, 1, 1]
        batch_target: torch.Tensor = input_dict[("color", 0, source_scale)]
        for scale in self.opt.scales:
            loss = 0.0
            reprojection_losses = []
            batch_disp: torch.Tensor = output_dict[("disp", scale)]
            batch_upscaled_depth: torch.Tensor = output_dict[("depth", scale)].squeeze(1)
            batch_color: torch.Tensor = input_dict[("color", 0, scale)]
            # batch_cam_pts: torch.Tensor = output_dict[("cam_pts", scale)]  # [bs, h, w, 3]
            batch_cam_pts = depths2cam_pts(batch_upscaled_depth, self.cam_grid)

            for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                batch_pred = output_dict[("color", adj_frame_idx, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(batch_pred, batch_target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for adj_frame_idx in self.opt.adj_frame_idxs[1:]:
                    batch_pred = input_dict[("color", adj_frame_idx, source_scale)]
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

            final_reprojection_loss = to_optimize.mean()
            loss_dict[f"loss/reprojection_{scale}"] = final_reprojection_loss.item()
            loss += final_reprojection_loss

            mean_disp = batch_disp.mean(2, True).mean(3, True)
            norm_disp = batch_disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, batch_color)

            loss_dict[f"loss/smoothness_{scale}"] = smooth_loss.item()
            loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)

            road_exist_idxs = batch_road.sum((1, 2)) > 0
            fine_metric_loss, unscaled_cam_heights = self.compute_cam_heights(
                batch_cam_pts[road_exist_idxs],
                batch_road[road_exist_idxs],
            )
            scaled_sum_cam_height_expects, scaled_sum_cam_height_vars = self.scale_cam_heights(
                batch_cam_pts[road_exist_idxs],
                batch_upscaled_depth[road_exist_idxs],
                batch_disp[road_exist_idxs],
                batch_segms[road_exist_idxs],
                batch_labels[road_exist_idxs],
                batch_n_insts[road_exist_idxs],
                batch_road[road_exist_idxs],
                unscaled_cam_heights,
                fy,
                scale,
            )

            if mode == "train" and scaled_sum_cam_height_expects is not None:
                self.sum_cam_height_expects_dict[scale] += scaled_sum_cam_height_expects
                self.sum_cam_height_vars_dict[scale] += scaled_sum_cam_height_vars

            if self.epoch > 0:
                loss_dict[f"loss/fine_metric_{scale}"] = fine_metric_loss.item()
                if self.opt.gradual_fine_metric_scale_weight:
                    rate = min(self.epoch / self.opt.increase_limit_epoch, 1)
                    loss += self.opt.fine_metric_scale_weight * rate * fine_metric_loss / (2**scale)
                else:
                    loss += self.opt.fine_metric_scale_weight * fine_metric_loss / (2**scale)

            total_loss += loss
            loss_dict[f"loss/{scale}"] = loss.item()

        total_loss /= self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict

    def compute_cam_heights(
        self,
        batch_cam_pts: torch.Tensor,  # [bs, h, w, 3]
        batch_road: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        batch_size = batch_cam_pts.shape[0]
        frame_unscaled_cam_heights = torch.zeros(batch_size, device=batch_cam_pts.device)
        loss = 0.0
        for batch_idx in range(batch_size):
            # cam_heights = cam_pts2cam_heights(batch_cam_pts[batch_idx][batch_road[batch_idx] == 1])  # [?, 3]
            cam_heights = cam_pts2cam_heights(batch_cam_pts[batch_idx], batch_road[batch_idx])  # [?, 3]
            if self.epoch > 0:
                loss = loss + F.gaussian_nll_loss(
                    input=self.prev_mean_cam_height_expects_dict[0],
                    target=cam_heights,
                    var=self.prev_mean_cam_height_vars_dict[0],
                    eps=0.001,
                    reduction="mean",
                )
            # TODO: 各点のcam_heightsをバッチでflattenしてからロスを計算する
            frame_unscaled_cam_heights[batch_idx] = cam_heights.detach().mean()
        if torch.isnan(loss).item():
            breakpoint()
        return loss / batch_size, frame_unscaled_cam_heights

    def scale_cam_heights(
        self,
        batch_cam_pts: torch.Tensor,  # [bs, h, w, 3]
        batch_depth: torch.Tensor,
        batch_disp: torch.Tensor,
        batch_segms: torch.Tensor,
        batch_labels: torch.Tensor,
        batch_n_insts: torch.Tensor,
        batch_road: torch.Tensor,
        frame_unscaled_cam_heights: torch.Tensor,
        fy: float,
        scale_idx: int,
    ) -> tuple[torch.Tensor, float, float]:
        if batch_n_insts.sum() == 0:
            return None, None

        _, h, w = batch_depth.shape

        # batch_eroded_segms = erode(batch_segms, self.kernel_size)
        # batch_n_insts = (batch_eroded_segms.sum((2, 3)) > 0).sum(1)
        # eroded_segms_flat = batch_eroded_segms.view(-1, h, w)
        # non_padded_channels = eroded_segms_flat.sum(dim=(1, 2)) > 0
        # segms_flat = batch_segms.view(-1, h, w)[non_padded_channels]  # exclude padded channels
        # eroded_segms_flat = eroded_segms_flat[non_padded_channels]
        # labels_flat = batch_labels.view(-1)[non_padded_channels].long()

        # depth_repeat = batch_depth.detach().repeat_interleave(batch_n_insts, dim=0)  # [sum(batch_n_insts), h, w]

        # obj_pix_heights = masks_to_pix_heights(segms_flat)
        # height_expects = self.height_priors[labels_flat, 0]
        # height_vars = self.height_priors[labels_flat, 1]
        # depth_expects = height_expects / obj_pix_heights * fy

        # cam_pts_repeat = batch_cam_pts.detach().repeat_interleave(batch_n_insts, dim=0)
        # masked_cam_pts = cam_pts_repeat * eroded_segms_flat.unsqueeze(-1)
        # masked_cam_pts[masked_cam_pts == 0] = 1000
        # nearest_pts = argmax_3d(-torch.linalg.norm(masked_cam_pts, dim=3))  # [n_insts * bs, 2]
        # # DONE: argmax_3dは正しい？-> おそらく問題ない
        # nearest_depths = depth_repeat[torch.arange(depth_repeat.shape[0]), nearest_pts[:, 0], nearest_pts[:, 1]]  # [n_insts * bs, 2]

        depth_repeat = batch_depth.detach().repeat_interleave(batch_n_insts, dim=0)  # [sum(batch_n_insts), h, w]
        segms_flat = batch_segms.view(-1, h, w)
        non_padded_channels = segms_flat.sum(dim=(1, 2)) > 0
        segms_flat = segms_flat[non_padded_channels]  # exclude padded channels
        labels_flat = batch_labels.view(-1)[non_padded_channels].long()

        obj_pix_heights = masks_to_pix_heights(segms_flat)
        height_expects = self.height_priors[labels_flat, 0]
        height_vars = self.height_priors[labels_flat, 1]
        depth_expects = height_expects * fy / obj_pix_heights

        cam_pts_repeat = batch_cam_pts.detach().repeat_interleave(batch_n_insts, dim=0)
        masked_cam_pts = cam_pts_repeat * segms_flat.unsqueeze(-1)
        masked_cam_pts[masked_cam_pts == 0] = 1000
        nearest_pts = argmax_3d(-torch.linalg.norm(masked_cam_pts, dim=3))  # [n_insts * bs, 2]
        nearest_depths = depth_repeat[torch.arange(depth_repeat.shape[0]), nearest_pts[:, 0], nearest_pts[:, 1]]  # [n_insts * bs, 2]

        batch_n_insts_lst = batch_n_insts.tolist()
        split_scale_expects = torch.split(depth_expects / nearest_depths, batch_n_insts_lst)
        frame_scale_expects = torch.tensor([chunk.mean() for chunk in split_scale_expects], device=depth_expects.device)
        # breakpoint()
        split_scale_vars = torch.split(height_vars / (obj_pix_heights * nearest_depths / fy) ** 2, batch_n_insts_lst)
        frame_scale_var = torch.tensor([chunk.sum() for chunk in split_scale_vars], device=height_vars.device) / batch_n_insts**2

        scaled_cam_height_expects = frame_scale_expects * frame_unscaled_cam_heights
        if scale_idx == 0:
            print(f"scaled_cam_height_expects: \n{scaled_cam_height_expects}")
            print(f"scale_expects: \n{frame_scale_expects}")
            print(f"unscaled_cam_heights: \n{frame_unscaled_cam_heights}\n")
            non_nan_idxs = ~frame_unscaled_cam_heights.isnan()
            if torch.allclose(
                frame_unscaled_cam_heights[non_nan_idxs],
                torch.full_like(non_nan_idxs[non_nan_idxs == True], 0.1, dtype=torch.float32),
                atol=0.0001,
                rtol=0,
            ):
                bs = batch_depth.shape[0]
                plt.figure(figsize=(15 * bs, 5 * 3))
                for idx in range(bs):
                    plt.subplot(3, bs, idx + 1)
                    plt.imshow(batch_segms.detach().cpu().numpy()[idx].sum(0))
                    plt.tight_layout()
                    plt.axis("off")
                for idx in range(bs, bs * 2):
                    plt.subplot(3, bs, idx + 1)
                    plt.imshow(batch_road.detach().cpu().numpy()[idx - bs])
                    plt.tight_layout()
                    plt.axis("off")
                for idx in range(bs * 2, bs * 3):
                    plt.subplot(3, bs, idx + 1)
                    plt.imshow(batch_disp.squeeze().detach().cpu().numpy()[idx - bs * 2], cmap="Spectral")
                    plt.tight_layout()
                    plt.axis("off")
                plt.show()
                plt.close()
            # breakpoint()
        scaled_sum_cam_height_expects = (frame_scale_expects * frame_unscaled_cam_heights).nansum()
        scaled_sum_cam_height_vars = (frame_scale_var * frame_unscaled_cam_heights**2).nansum()
        return scaled_sum_cam_height_expects, scaled_sum_cam_height_vars

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
            print(f"Loading {model_name} weights...")
            path = weights_dir / f"{model_name}.pth"
            model_dict = self.models[model_name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[model_name].load_state_dict(model_dict)

        # load prev_cam_height
        cam_height_expect_path = weights_dir / "cam_height_expect.pkl"
        if cam_height_expect_path.exists():
            print("Loading prev_mean_cam_height_expects_dict")
            with open(cam_height_expect_path, "rb") as f:
                self.prev_mean_cam_height_expects_dict = pickle.load(f)
        else:
            print(f"\n{cam_height_expect_path} does not exists.")

        cam_height_var_path = weights_dir / "cam_height_var.pkl"
        if cam_height_var_path.exists():
            print("Loading prev_mean_cam_height_vars_dict")
            with open(cam_height_var_path, "rb") as f:
                self.prev_mean_cam_height_vars_dict = pickle.load(f)
        else:
            print(f"\n{cam_height_var_path} does not exists.")

        # load adam state
        optimizer_load_path = weights_dir / "adam.pth"
        if optimizer_load_path.exists():
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
