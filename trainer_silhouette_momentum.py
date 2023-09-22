import warnings

warnings.filterwarnings("ignore")

import time

import torch

from layers import get_smooth_loss
from trainer_silhouette_base import TrainerSilhouetteBase
from utils import (
    calc_obj_pix_height_over_dist_to_horizon,
    cam_pts2normal,
    compute_scaled_cam_heights,
    sigmoid,
)


class TrainerSilhouetteMomentum(TrainerSilhouetteBase):
    def __init__(self, options):
        if options.sparse_update:
            raise ValueError("This class does not support --sparse_update")
        if options.wo_1st_update:
            raise ValueError("This class does not support --wo_1st_update")

        super().__init__(options)
        self.scale_init_dict = {scale: torch.full((len(self.train_loader.dataset),), torch.nan) for scale in self.opt.scales}
        self.whole_cam_heights_dict: dict[int, torch.Tensor] = self.scale_init_dict.copy()

    def train(self):
        """Run the entire training pipeline"""
        self.step = self.epoch * len(self.train_loader)
        self.start_time = time.time()
        lower_is_better = self.standard_metric[:2] == "de" or self.standard_metric == "loss"
        th_best = 1000000 if lower_is_better else -1

        print(f"========= Training has started from {self.epoch} epoch. ========= ")
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.n_inst_frames = 0
            self.whole_cam_heights_dict = self.scale_init_dict.copy()
            self.cam_height_idx_last = 0

            self.train_epoch()
            val_loss_dict = self.val_epoch()
            val_loss = val_loss_dict[self.standard_metric]
            new_cam_height_dict = {
                scale: torch.nanquantile(whole_cam_heights, q=0.5) for scale, whole_cam_heights in self.whole_cam_heights_dict.items()
            }

            if self.epoch == 0:
                # train_epoch()ではインスタンスがprev_cam_height_dictをattrとして持っているかで分岐する処理があるのでここで初期化
                self.prev_cam_height_dict = {scale: 0.0 for scale in self.opt.scales}
            epoch = self.epoch + 1
            self.prev_cam_height_dict = {
                scale: (prev_cam_height * (epoch - 1) * epoch / 2 + epoch * cam_height) / (epoch * (epoch + 1) / 2)
                for (scale, prev_cam_height), cam_height in zip(self.prev_cam_height_dict.items(), new_cam_height_dict.values())
            }
            if not self.opt.dry_run:
                self.log_cam_height()
                if (lower_is_better and val_loss < th_best) or (not lower_is_better and val_loss > th_best):
                    self.save_model(is_best=True)
                    th_best = val_loss
                else:
                    self.save_model(is_best=False)

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
            segms_flat, obj_pix_heights, obj_height_expects = self.make_flats(batch_segms, batch_labels)

            if self.opt.remove_outliers and batch_road_wo_no_road.shape[0] > 0 and hasattr(self, "prev_cam_height_dict"):
                bs_wo_no_road = batch_road_wo_no_road.shape[0]
                batch_cam_pts_wo_no_road: torch.Tensor = output_dict[("cam_pts", source_scale)].detach()[batch_road_appear_bools]
                normals = torch.zeros((bs_wo_no_road, 3, 1), device=self.device)
                for batch_wo_no_road_idx in range(bs_wo_no_road):
                    normals[batch_wo_no_road_idx] = cam_pts2normal(
                        batch_cam_pts_wo_no_road[batch_wo_no_road_idx], batch_road_wo_no_road[batch_wo_no_road_idx]
                    )
                cam_height = self.prev_cam_height_dict[0]
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

                split_inlier_mask = torch.split(inlier_bools, batch_n_insts.tolist())
                batch_n_insts = torch.tensor([chunk.sum() for chunk in split_inlier_mask], device=self.device)
                n_inst_appear_frames = (batch_n_insts > 0).sum()

            # TODO: 道路が無いときはどの物体も信頼すべきではないので取り除くべきでは？現状道路がない＝物体もない環境がほとんどなので大差はなさそう

        if mode == "train":
            self.n_inst_frames += (batch_n_insts[batch_road_appear_bools] > 0).sum()

        fy: float = input_dict[("K", source_scale)][0, 1, 1]
        batch_target: torch.Tensor = input_dict[("color", 0, source_scale)]
        for scale_idx, scale in enumerate(self.opt.scales):
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

            final_reprojection_loss = to_optimize.mean()
            loss_dict[f"loss/reprojection_{scale}"] = final_reprojection_loss.item()
            loss += final_reprojection_loss

            mean_disp = batch_disp.mean(2, True).mean(3, True)
            norm_disp = batch_disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, batch_color)

            loss_dict[f"loss/smoothness_{scale}"] = smooth_loss.item()
            loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)

            if batch_road_wo_no_road.shape[0] > 0:
                fine_metric_loss, normal_loss, unscaled_cam_heights, road_normal_neg_wo_no_road = self.compute_cam_heights(
                    batch_cam_pts_wo_no_road,
                    batch_road_wo_no_road,
                )
                if self.opt.with_normal_loss:
                    loss_dict[f"loss/normal_{scale}"] = normal_loss.item()
                    if self.opt.gradual_normal_weight:
                        loss += fine_metric_rate * normal_loss
                    else:
                        loss += normal_loss
                if hasattr(self, "prev_cam_height_dict"):
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
                    fy,
                    n_inst_appear_frames,
                )
                loss_dict[f"loss/rough_metric_{scale}"] = rough_metric_loss.item()
                loss += self.opt.rough_metric_scale_weight * rough_metric_rate * rough_metric_loss

                if batch_road_wo_no_road.shape[0] > 0:
                    flat_road_appear_idxs = batch_road_appear_bools.repeat_interleave(batch_n_insts, dim=0)
                    scaled_cam_heights = compute_scaled_cam_heights(
                        segms_flat[flat_road_appear_idxs],
                        batch_n_insts[batch_road_appear_bools],
                        road_normal_neg_wo_no_road,
                        batch_cam_pts_wo_no_road.detach(),
                        unscaled_cam_heights,
                        obj_height_expects[flat_road_appear_idxs],
                        from_ground=self.opt.from_ground,
                    )
                    if mode == "train":
                        new_idx_last = self.cam_height_idx_last + scaled_cam_heights.shape[0]
                        self.whole_cam_heights_dict[scale][self.cam_height_idx_last : new_idx_last] = scaled_cam_heights
                        if scale_idx == len(self.opt.scales) - 1:
                            self.cam_height_idx_last = new_idx_last

            total_loss += loss
            loss_dict[f"loss/{scale}"] = loss.item()

        total_loss /= self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict
