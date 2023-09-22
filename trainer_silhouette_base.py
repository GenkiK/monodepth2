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
from layers import SSIM, BackprojectDepth, Project3D, compute_depth_errors, disp_to_depth, transformation_from_parameters
from utils import (
    cam_pts2cam_height_with_cross_prod,
    generate_homo_pix_grid,
    masks_to_pix_heights,
    normalize_image,
    readlines,
    sec_to_hm_str,
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


class TrainerSilhouetteBase:
    def __init__(self, options):
        self.opt = options
        if self.opt.sparse_update and self.opt.update_freq is None:
            raise ValueError("When --sparse_update is True, specify --update_freq.")

        err_msg = "This script doesn't support "
        if self.opt.damping_update:
            raise ValueError(err_msg + "--damping_update.")
        if self.opt.wo_1st2nd_update:
            raise ValueError(err_msg + "--wo_1st2nd_update.")
        if self.opt.warmup:
            raise ValueError(err_msg + "--warmup.")
        if self.opt.soft_remove_outliers:
            raise ValueError(err_msg + "--soft_remove_outliers.")

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
            print(f"Initial cam_height is {self.prev_cam_height_dict[0]}\n")

        if can_resume:
            if self.opt.last_epoch_for_resume is None:
                weights_dir, self.epoch = self.search_last_epoch()
            else:
                weights_dir = Path(self.log_path) / "models" / f"weights_{self.opt.last_epoch_for_resume}"
                self.epoch = self.opt.last_epoch_for_resume
            self.epoch += 1
            self.load_model(weights_dir)
            self.load_cam_heights(weights_dir, False)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, self.opt.gamma)
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
            self.height_priors = TrainerSilhouetteBase.read_height_priors(self.opt.data_path)
            print("\nUsing calculated object height.\n")
        self.height_priors = self.height_priors.to(self.device)
        self.homo_pix_grid = generate_homo_pix_grid(self.opt.height, self.opt.width)

        self.outlier_th = self.opt.outlier_relative_error_th
        if not self.opt.dry_run:
            self.save_opts()

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def compute_losses(self, input_dict, output_dict, mode):
        raise NotImplementedError("compute_losses() is not implemented.")

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

    def train_epoch(self):
        """Run a single epoch of training and validation"""
        print("Training")
        self.set_train()

        for iteration, batch_input_dict in tqdm(enumerate(self.train_loader), dynamic_ncols=True):
            before_op_time = time.time()

            batch_input_dict = {key: ipt.to(self.device) for key, ipt in batch_input_dict.items()}
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

    def compute_cam_heights(
        self,
        batch_cam_pts: torch.Tensor,  # [bs, h, w, 3]
        batch_road: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = batch_cam_pts.shape[0]
        frame_unscaled_cam_heights = torch.zeros(bs, device=self.device)
        cam_height_loss = torch.tensor(0.0, device=self.device)
        normal_loss = torch.tensor(0.0, device=self.device)
        batch_road_normal_neg = torch.zeros((bs, 3), device=self.device)
        batch_cam_height, batch_normal = cam_pts2cam_height_with_cross_prod(batch_cam_pts)
        batch_road[:, -1, :] = 0
        batch_road[:, :, 0] = 0
        batch_road[:, :, -1] = 0
        for batch_idx in range(bs):
            road_cam_heights = batch_cam_height[batch_idx, batch_road[batch_idx] == 1]  # [?,]
            normals = batch_normal[batch_idx, batch_road[batch_idx] == 1]  # [?, 3]
            sum_normals = normals.detach().sum(0)
            batch_road_normal_neg[batch_idx] = -sum_normals / torch.norm(sum_normals)
            normal_loss = normal_loss + torch.abs(normals - normals.detach().quantile(0.5, dim=0)).mean(0).mean()
            if hasattr(self, "prev_cam_height_dict"):
                match self.opt.cam_height_loss_func:
                    case "abs":
                        cam_height_loss = cam_height_loss + torch.abs(self.prev_cam_height_dict[0] - road_cam_heights).mean()
            if self.opt.use_median_cam_height:
                frame_unscaled_cam_heights[batch_idx] = road_cam_heights.detach().quantile(0.5)
            else:
                frame_unscaled_cam_heights[batch_idx] = road_cam_heights.detach().mean()
        return cam_height_loss / bs, normal_loss / bs, frame_unscaled_cam_heights, batch_road_normal_neg

    def compute_rough_metric_loss(
        self,
        depth_repeat: torch.Tensor,
        segms_flat: torch.Tensor,
        obj_pix_heights: torch.Tensor,
        obj_height_expects: torch.Tensor,
        fy: float,
        n_inst_appear_frames: int,
    ) -> torch.Tensor:
        match self.opt.rough_metric_loss_func:
            case "abs":
                obj_mean_depths = (depth_repeat * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2)).clamp(min=1e-9)
                pred_heights = obj_pix_heights * obj_mean_depths / fy
                loss = torch.abs(obj_height_expects - pred_heights)
            case "mean_after_abs":
                pred_heights = obj_pix_heights[:, None, None] * depth_repeat / fy  # [sum(batch_n_insts), h, w]
                loss = torch.abs((obj_height_expects[:, None, None] - pred_heights) * segms_flat).sum(dim=(1, 2)) / segms_flat.sum(dim=(1, 2))
        assert not loss.mean().isnan()
        return loss.mean() / n_inst_appear_frames

    def make_flats(
        self,
        batch_segms: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h, w = batch_segms.shape
        segms_flat = batch_segms.view(-1, h, w)
        non_padded_channels = segms_flat.sum(dim=(1, 2)) > 0
        segms_flat = segms_flat[non_padded_channels]  # exclude padded channels
        labels_flat = batch_labels.view(-1)[non_padded_channels].long()

        obj_pix_heights = masks_to_pix_heights(segms_flat)
        obj_height_expects = self.height_priors[labels_flat, 0]
        return segms_flat, obj_pix_heights, obj_height_expects

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
                writer.add_scalar(f"cam_height_expect_{scale}", self.prev_cam_height_dict[scale], self.step)
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

        if hasattr(self, "prev_cam_height_dict"):
            cam_height_expect_path = os.path.join(save_folder, "cam_height_expect.pkl")
            with open(cam_height_expect_path, "wb") as f:
                pickle.dump(self.prev_cam_height_dict, f, pickle.HIGHEST_PROTOCOL)

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
            print("Loading prev_cam_height_dict")
            with open(cam_height_expect_path, "rb") as f:
                self.prev_cam_height_dict = pickle.load(f)
        elif alert_if_not_exist:
            raise FileNotFoundError(f"\n{cam_height_expect_path} does not exists.\n")
        else:
            print(f"\n{cam_height_expect_path} does not exists.\n")

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
