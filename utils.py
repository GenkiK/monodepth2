# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import hashlib
import os
import random
import urllib
import zipfile
from math import exp

import numpy as np
import torch
from torch.nn import functional as F

LARGE_VALUE = 100


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


def sigmoid(x: int | float) -> float:
    return 1 / (1 + exp(-x))


def calc_projected_heights(
    segms: torch.Tensor, road_normal_neg: torch.Tensor, cam_pts: torch.Tensor, unscaled_cam_height: torch.float32, from_ground: bool = False
) -> torch.Tensor:
    """
    segms: [n_inst, h, w], cuda
    road_normal_neg: [3], cuda
    cam_pts: [h, w, 3], cuda
    cam_height: torch.float32
    """
    device = segms.device
    n_inst = segms.shape[0]
    nx, ny, _ = road_normal_neg
    origin = torch.tensor((0, unscaled_cam_height / ny, 0), dtype=torch.float32, device=device)
    root = torch.sqrt(ny**2 / (nx**2 + ny**2))
    x_basis = torch.tensor((root, -nx / ny * root, 0), dtype=torch.float32, device=device)
    z_basis = torch.cross(x_basis, road_normal_neg)
    projected_cam_pts = cam_pts - z_basis[None, None, :] * (cam_pts @ z_basis).unsqueeze(-1)
    projected_cam_pts -= origin[None, None, :]  # [h, w, 3]
    ys = projected_cam_pts @ road_normal_neg  #
    projected_heights = torch.zeros((n_inst,), dtype=torch.float32, device=device)
    if from_ground:
        for idx in range(n_inst):
            if (segm_max := ys[segms[idx]].max()) < 0:
                projected_heights[idx] = -ys[segms[idx]].min()
            else:
                projected_heights[idx] = segm_max - ys[segms[idx]].min()
    else:
        for idx in range(n_inst):
            projected_heights[idx] = ys[segms[idx]].max() - ys[segms[idx]].min()
    return projected_heights


def compute_scaled_sum_cam_height(
    segms_flat: torch.Tensor,
    batch_n_inst: torch.Tensor,
    batch_road_normal_neg: torch.Tensor,
    batch_cam_pts: torch.Tensor,
    batch_unscaled_cam_height: torch.Tensor,
    obj_height_expects: torch.Tensor,
    from_ground: bool = False,
) -> torch.Tensor:
    """
    segms_flat: [n_inst, h, w], cuda
    batch_road_normal_neg: [bs, 3], cuda
    batch_cam_pts: [bs, h, w, 3], cuda
    batch_unscaled_cam_height: [bs,], cuda
    """
    device = segms_flat.device
    bs = batch_cam_pts.shape[0]
    nx, ny, _ = batch_road_normal_neg.T
    batch_origin = torch.stack((torch.zeros(bs, device=device), batch_unscaled_cam_height / ny, torch.zeros(bs, device=device)), dim=1)  # [bs, 3]
    batch_root = torch.sqrt(ny**2 / (nx**2 + ny**2))
    batch_x_basis = torch.stack((batch_root, -ny / ny * batch_root, torch.zeros(bs, device=device)), dim=1)  # [bs, 3]
    batch_z_basis = torch.cross(batch_x_basis, batch_road_normal_neg)  # [bs, 3]
    projected_cam_pts = batch_cam_pts - batch_z_basis[:, None, None, :] * torch.einsum("bijk,bk->bij", batch_cam_pts, batch_z_basis).unsqueeze(-1)
    projected_cam_pts -= batch_origin[:, None, None, :]  # [bs, h, w, 3]
    batch_ys = torch.einsum("bijk,bk->bij", projected_cam_pts, batch_road_normal_neg)  # [bs, h, w]

    projected_height_flat = torch.zeros((segms_flat.shape[0],), device=device)
    prev_n_inst = 0
    if from_ground:
        for batch_idx, n_inst in enumerate(batch_n_inst):
            for idx in range(prev_n_inst, prev_n_inst + n_inst):
                region = batch_ys[batch_idx, segms_flat[idx]]
                if (segm_max := region.max()) < 0:
                    projected_height_flat[idx] = -region.min()
                else:
                    projected_height_flat[idx] = segm_max - region.min()
            prev_n_inst += n_inst
    else:
        for batch_idx, n_inst in enumerate(batch_n_inst):
            for idx in range(prev_n_inst, prev_n_inst + n_inst):
                region = batch_ys[batch_idx, segms_flat[idx]]
                projected_height_flat[idx] = region.max() - region.min()
            prev_n_inst += n_inst
    split_scale_expects = torch.split(obj_height_expects / projected_height_flat, batch_n_inst.tolist())
    frame_scale_expects = torch.tensor([chunk.quantile(0.5) if chunk.numel() > 0 else torch.nan for chunk in split_scale_expects], device=device)
    scaled_sum_cam_height = (frame_scale_expects * batch_unscaled_cam_height).nansum()
    return scaled_sum_cam_height


def calc_occluded_obj_pix_height(
    homo_pix_grid: torch.Tensor,
    batch_segms: torch.Tensor,
    # batch_obj_height_expects: torch.Tensor,
    obj_height_expects_flat: torch.Tensor,
    cam_height: float,
    horizons: torch.Tensor,
    batch_n_insts: torch.Tensor,
    batch_road_appear_bools: torch.Tensor,
) -> torch.Tensor:
    """
    道路が無いときを除いた，画像平面における物体の上底と下底の候補点．
    道路が見えない状況は少ない＆そのような状況では物体が少ない という理由から道路が見えてないときは全て無視する方針にする

    Args:
        homo_pix_grid (torch.Tensor): [3, h, w]
        batch_segms (torch.Tensor): [bs, max_n_inst, h, w]
        obj_height_expects_flat: [n_inst in batch, h, w]
        cam_height: float
        horizons (torch.Tensor): [bs_wo_no_road, 3], detached
        batch_n_insts (torch.Tensor): [bs,]
        batch_road_appear_bools: [bs,]

    Returns:
        torch.Tensor: top_pos_wo_no_road_flat
        torch.Tensor: bottom_pos_wo_no_road_flat
    """
    _, _, h, w = batch_segms.shape
    device = batch_segms.device
    batch_A, batch_B = horizon_to_2pts(horizons)
    batch_AB = batch_B - batch_A  # [bs_wo_no_road, 2]
    norm_AB = torch.norm(batch_AB, dim=1)  # [bs_wo_no_road,]
    batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=device)), dim=1)  # add zeros for torch.cross -> [bs_wo_no_road, 3]
    dim_y_eraser = torch.tensor([1, 0, 1], dtype=torch.uint8, device=device)[:, None]
    homo_pix_grid = homo_pix_grid.to(device)
    batch_orthogonal_vecs = batch_vec_orthogonal_to_horizon(horizons[:, :2])  # [bs_wo_no_road, 2]

    bs_wo_no_road = horizons.shape[0]
    # homo_pix_grid.view(3, -1)[:2]: [2, h*w], batch_A: [bs_wo_no_road, 2]
    batch_AP = homo_pix_grid.view(3, -1)[:2, :].unsqueeze(0) - batch_A.unsqueeze(-1)  # [bs_wo_no_road, 2, h*w]
    batch_AP = torch.cat((batch_AP, torch.zeros((bs_wo_no_road, 1, h * w), device=device)), dim=1)  # [bs_wo_no_road, 3, h*w]
    # batch_AB: [bs_wo_no_road, 3],
    batch_cross = torch.linalg.cross(batch_AB.unsqueeze(-1), batch_AP, dim=1)[:, -1, :]  # [bs_wo_no_road, h*w]
    batch_dist = (torch.abs(batch_cross) / norm_AB[:, None]).view(bs_wo_no_road, h, w)

    # each pos is [y, x]
    total_n_inst_wo_no_road = batch_n_insts[batch_road_appear_bools].sum()
    bottom_pos_wo_no_road_flat = torch.zeros((total_n_inst_wo_no_road, 2), dtype=torch.int16, device=device)
    top_pos_wo_no_road_flat = torch.zeros((total_n_inst_wo_no_road, 2), dtype=torch.int16, device=device)
    color_idx_flat = [0] * total_n_inst_wo_no_road

    flat_idx = 0
    batch_idx_wo_no_road = -1
    for batch_idx in range(batch_segms.shape[0]):
        if batch_road_appear_bools[batch_idx]:
            batch_idx_wo_no_road += 1
        else:
            continue
        orthogonal_vec = batch_orthogonal_vecs[batch_idx_wo_no_road]
        for inst_idx in range(batch_n_insts[batch_road_appear_bools][batch_idx_wo_no_road]):
            dist = batch_dist[batch_idx_wo_no_road, batch_segms[batch_idx, inst_idx]]  # [bs, n_inst_pix]
            homo_inst_pix_pos = homo_pix_grid[:, batch_segms[batch_idx, inst_idx]]  # [3, n_inst_pix]
            inst_pix_pos = homo_inst_pix_pos[:2]  # [2, n_inst_pix]
            # TODO: 一応upper_region_maskが合っているか確認する
            upper_region_mask = inst_pix_pos[1] < (
                -(horizons[batch_idx_wo_no_road] @ (homo_inst_pix_pos * dim_y_eraser)) / horizons[batch_idx_wo_no_road, 1]
            )  # [n_inst_pix,]

            obj_height_expect = obj_height_expects_flat[flat_idx]
            upper_area = upper_region_mask.sum()
            lower_area = (~upper_region_mask).sum()
            if upper_area > 0:
                pix_diff, i = dist[upper_region_mask].max(dim=0)
                signed_pix_diff = -pix_diff
            else:
                signed_pix_diff, i = dist.min(dim=0)
            top_pos = torch.nonzero(batch_segms[batch_idx, inst_idx])[i]

            """
            課題

            * horizonが間違ってた時に結構コケる．また現在のhorizonに合わないものはどんどん取り除かれていってしまう
            *
            *
            """

            if signed_pix_diff > 0 and cam_height > obj_height_expect:  # ①
                segm_height = dist[~upper_region_mask].max() - signed_pix_diff
                # obj_pix_height = max(segm_height, obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff)  # FIXME
                # TODO: ↑差が大きすぎる時にはsegm_heightを使うっていう方針にしても良いかも（でもそれ結果的に毎回segm_heightを使うことに繋がりそう）

                obj_pix_height = obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff
                # obj_pix_height = segm_height
                color_idx = 0  # dark green
            elif signed_pix_diff < 0 and cam_height < obj_height_expect:  # ①
                if lower_area > 0:
                    segm_height = dist[~upper_region_mask].max() - signed_pix_diff
                else:  # すべてhorizonより上またはhorizonの線上
                    segm_height = -signed_pix_diff - dist[upper_region_mask].min()
                # obj_pix_height = max(segm_height, obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff) # FIXME
                obj_pix_height = obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff
                # obj_pix_height = segm_height
                color_idx = 1  # orange
            elif cam_height == obj_height_expect:  # ②
                if lower_area > 0:
                    obj_pix_height = dist[~upper_region_mask].max() - signed_pix_diff
                else:
                    obj_pix_height = -signed_pix_diff - dist[upper_region_mask].min()
                color_idx = 2  # purple
            elif signed_pix_diff == 0:  # ③
                margin = +1 if cam_height > obj_height_expect else -1
                signed_pix_diff += margin
                top_pos[0] += margin
                if signed_pix_diff > 0 and cam_height > obj_height_expect:  # ①
                    segm_height = dist[~upper_region_mask].max() - signed_pix_diff
                    # obj_pix_height = max(segm_height, obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff) # FIXME
                    obj_pix_height = obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff
                    # obj_pix_height = segm_height
                    color_idx = 3  # pink
                elif signed_pix_diff < 0 and cam_height < obj_height_expect:  # ①
                    if lower_area > 0:
                        segm_height = dist[~upper_region_mask].max() - signed_pix_diff
                    else:
                        segm_height = -signed_pix_diff - dist[upper_region_mask].min()
                    # obj_pix_height = max(segm_height, obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff) # FIXME
                    obj_pix_height = obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff
                    # obj_pix_height = segm_height
                    color_idx = 4  # green
            else:
                if lower_area > 0:
                    segm_height = dist[~upper_region_mask].max() - signed_pix_diff
                else:
                    segm_height = -signed_pix_diff - dist[upper_region_mask].min()
                # obj_pix_height = max(segm_height, torch.abs(obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff)) # FIXME
                obj_pix_height = torch.abs(obj_height_expect / (cam_height - obj_height_expect) * signed_pix_diff)
                # obj_pix_height = segm_height
                color_idx = 5  # yellow

            bottom_pos_wo_no_road_flat[flat_idx] = (top_pos + (orthogonal_vec * obj_pix_height).flip(0)).to(torch.int16)  # Tensor([y, x])
            top_pos_wo_no_road_flat[flat_idx] = top_pos  # Tensor([y, x])
            color_idx_flat[flat_idx] = color_idx
            flat_idx += 1
    return top_pos_wo_no_road_flat, bottom_pos_wo_no_road_flat, color_idx_flat


def batch_vec_orthogonal_to_horizon(batch_horizon_vec: torch.Tensor) -> torch.Tensor:
    """
    horizon_vec: torch.Size([bs, 2]) [x, y]

    return: torch.Size([bs, 2]) [x, y]
    """
    batch_norm = batch_horizon_vec.norm(dim=1)
    batch_vec = batch_horizon_vec / batch_norm[:, None]
    return batch_vec * ((batch_vec[:, 1] > 0) * 2 - 1)[:, None]  # y方向が正になるように正負を変換


def calc_obj_pix_height_over_dist_to_horizon(
    # HACK: 道路がない画像のインスタンスのratioは0にしてる（＝取り除かない）けど，OK?
    homo_pix_grid: torch.Tensor,
    batch_segms: torch.Tensor,
    horizons: torch.Tensor,
    batch_n_insts: torch.Tensor,
    batch_road_appear_bools: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        homo_pix_grid (torch.Tensor): [3, h, w]
        batch_segms (torch.Tensor): [bs, max_n_inst, h, w]
        horizons (torch.Tensor): [bs_wo_no_road, 3], detached
        batch_n_insts (torch.Tensor): [bs,]
        batch_road_appear_bools: [bs,]

    Returns:
        torch.Tensor: _description_
    """
    _, _, h, w = batch_segms.shape
    device = batch_segms.device
    total_n_inst = batch_n_insts.sum()
    ratios = torch.zeros((total_n_inst,), dtype=torch.float32, device=device)
    batch_A, batch_B = horizon_to_2pts(horizons)
    batch_AB = batch_B - batch_A  # [bs_wo_no_road, 2]
    norm_AB = torch.norm(batch_AB, dim=1)  # [bs_wo_no_road,]
    batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=device)), dim=1)  # add zeros for torch.cross -> [bs_wo_no_road, 3]
    dim_y_eraser = torch.tensor([1, 0, 1], dtype=torch.uint8, device=device)[:, None]
    homo_pix_grid = homo_pix_grid.to(device)
    inst_idx_in_batch = 0
    ratio_idxs = torch.where(batch_road_appear_bools.repeat_interleave(batch_n_insts, dim=0))[0]

    bs_wo_no_road = horizons.shape[0]
    # homo_pix_grid.view(3, -1)[:2]: [2, h*w], batch_A: [bs_wo_no_road, 2]
    batch_AP = homo_pix_grid.view(3, -1)[:2, :].unsqueeze(0) - batch_A.unsqueeze(-1)  # [bs_wo_no_road, 2, h*w]
    batch_AP = torch.cat((batch_AP, torch.zeros((bs_wo_no_road, 1, h * w), device=device)), dim=1)  # [bs_wo_no_road, 3, h*w]
    # batch_AB: [bs_wo_no_road, 3],
    batch_cross = torch.linalg.cross(batch_AB.unsqueeze(-1), batch_AP, dim=1)[:, -1, :]  # [bs_wo_no_road, h*w]
    batch_dist = (torch.abs(batch_cross) / norm_AB[:, None]).view(bs_wo_no_road, h, w)

    batch_idx_wo_no_road = -1
    for batch_idx in range(batch_segms.shape[0]):
        if batch_road_appear_bools[batch_idx]:
            batch_idx_wo_no_road += 1
        else:
            continue
        for inst_idx in range(batch_n_insts[batch_road_appear_bools][batch_idx_wo_no_road]):
            dist = batch_dist[batch_idx_wo_no_road, batch_segms[batch_idx, inst_idx] == 1]
            homo_valid_pos = homo_pix_grid[:, batch_segms[batch_idx, inst_idx] == 1]  # [3, n_valid_pos]
            valid_pos = homo_valid_pos[:2]  # [2, n_valid_pos]
            upper_region_mask = valid_pos[1] < (
                -(horizons[batch_idx_wo_no_road] @ (homo_valid_pos * dim_y_eraser)) / horizons[batch_idx_wo_no_road, 1]
            )  # [n_valid_pos,]
            if (~upper_region_mask).sum() == 0:
                ratios[ratio_idxs[inst_idx_in_batch]] = LARGE_VALUE
            elif upper_region_mask.sum() > 0:
                dist_to_horizon = dist[~upper_region_mask].max()
                obj_pix_height = dist_to_horizon + dist[upper_region_mask].max()
                ratios[ratio_idxs[inst_idx_in_batch]] = obj_pix_height / dist_to_horizon
            else:
                dist_to_horizon = dist.max()
                obj_pix_height = dist_to_horizon - dist.min()
                ratios[ratio_idxs[inst_idx_in_batch]] = obj_pix_height / dist_to_horizon
            inst_idx_in_batch += 1
    return ratios


# def calc_obj_pix_height_over_dist_to_horizon(
#     # HACK: 道路がない画像のインスタンスのratioは0にしてる（＝取り除かない）けど，OK?
#     homo_pix_grid: torch.Tensor,
#     batch_segms: torch.Tensor,
#     horizons: torch.Tensor,
#     batch_n_insts: torch.Tensor,
#     batch_road_appear_bools: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Args:
#         homo_pix_grid (torch.Tensor): [3, h, w]
#         batch_segms (torch.Tensor): [bs, max_n_inst, h, w]
#         horizons (torch.Tensor): [bs_wo_no_road, 3], detached
#         batch_n_insts (torch.Tensor): [bs,]
#         batch_road_appear_bools: [bs,]

#     Returns:
#         torch.Tensor: _description_
#     """
#     device = batch_segms.device
#     total_n_inst = batch_n_insts.sum()
#     ratios = torch.zeros((total_n_inst,), dtype=torch.float32, device=device)
#     batch_A, batch_B = horizon_to_2pts(horizons)
#     batch_AB = batch_B - batch_A
#     norm_AB = torch.norm(batch_AB, dim=1)  # [bs,]
#     batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=device)), dim=1)  # add zeros for torch.cross
#     dim_y_eraser = torch.tensor([1, 0, 1], dtype=torch.uint8, device=device)[:, None]
#     homo_pix_grid = homo_pix_grid.to(device)
#     inst_idx_in_batch = 0
#     ratio_idxs = torch.where(batch_road_appear_bools.repeat_interleave(batch_n_insts, dim=0))[0]
#     for batch_idx in range(horizons.shape[0]):
#         for inst_idx in range(batch_n_insts[batch_road_appear_bools][batch_idx]):
#             homo_valid_pos = homo_pix_grid[:, batch_segms[batch_idx, inst_idx] == 1]  # [3, n_valid_pos]
#             valid_pos = homo_valid_pos[:2]  # [2, n_valid_pos]
#             APs = valid_pos - batch_A[batch_idx, :, None]
#             n_pts = APs.shape[1]
#             APs = torch.cat((APs, torch.zeros((1, n_pts), device=device)), dim=0)  # add zeros for torch.cross
#             cross = torch.linalg.cross(batch_AB[batch_idx][:, None].repeat((1, n_pts)), APs, dim=0)[-1]
#             dist = torch.abs(cross) / norm_AB[batch_idx]
#             upper_region_mask = valid_pos[1] < (-(horizons[batch_idx] @ (homo_valid_pos * dim_y_eraser)) / horizons[batch_idx, 1])  # [n_valid_pos,]
#             if (~upper_region_mask).sum() == 0:
#                 ratios[ratio_idxs[inst_idx_in_batch]] = LARGE_VALUE
#             elif upper_region_mask.sum() > 0:
#                 dist_to_horizon = dist[~upper_region_mask].max()
#                 obj_pix_height = dist_to_horizon + dist[upper_region_mask].max()
#                 ratios[ratio_idxs[inst_idx_in_batch]] = obj_pix_height / dist_to_horizon
#             else:
#                 dist_to_horizon = dist.max()
#                 obj_pix_height = dist_to_horizon - dist.min()
#                 ratios[ratio_idxs[inst_idx_in_batch]] = obj_pix_height / dist_to_horizon
#             inst_idx_in_batch += 1
#     return ratios


# def calc_obj_pix_height_over_dist_to_horizon(
#     homo_pix_grid: torch.Tensor, segms_flat: torch.Tensor, horizons: torch.Tensor, batch_n_insts: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Args:
#         homo_pix_grid (torch.Tensor): [3, h, w]
#         segms_flat (torch.Tensor): [n_inst in batch, h, w]
#         horizons (torch.Tensor): [bs, 3], detached
#         batch_n_insts (torch.Tensor): [bs,]

#     Returns:
#         torch.Tensor: _description_
#     """
#     device = segms_flat.device
#     n_inst = segms_flat.shape[0]
#     ratios = torch.zeros((n_inst,), dtype=torch.float32, device=device)
#     batch_A, batch_B = horizon_to_2pts(horizons)
#     batch_AB = batch_B - batch_A
#     norm_AB = torch.norm(batch_AB, dim=1)  # [bs,]
#     batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=device)), dim=1)  # add zeros for torch.cross
#     batch_A = batch_A.repeat_interleave(batch_n_insts, dim=0)
#     batch_AB = batch_AB.repeat_interleave(batch_n_insts, dim=0)
#     norm_AB = norm_AB.repeat_interleave(batch_n_insts, dim=0)
#     dim_y_eraser = torch.tensor([1, 0, 1], dtype=torch.uint8, device=device)[:, None]
#     homo_pix_grid = homo_pix_grid.to(device)
#     for i in range(n_inst):
#         homo_valid_pos = homo_pix_grid[:, segms_flat[i] == 1]  # [3, n_valid_pos]
#         valid_pos = homo_valid_pos[:2]  # [2, n_valid_pos]
#         APs = valid_pos - batch_A[i, :, None]
#         n_pts = APs.shape[1]
#         APs = torch.cat((APs, torch.zeros((1, n_pts), device=device)), dim=0)  # add zeros for torch.cross
#         cross = torch.linalg.cross(batch_AB[i][:, None].repeat((1, n_pts)), APs, dim=0)[-1]
#         dist = torch.abs(cross) / norm_AB[i]
#         # dist = torch.abs(torch.cross(batch_AB[i : i + 1], APs, dim=0)) / norm_AB[i]
#         upper_region_mask = valid_pos[1] < (-(horizons[i] @ (homo_valid_pos * dim_y_eraser)) / horizons[i, 1])  # [n_valid_pos,]
#         if (~upper_region_mask).sum() == 0:
#             ratios[i] = LARGE_VALUE
#         elif upper_region_mask.sum() > 0:
#             dist_to_horizon = dist[~upper_region_mask].max()
#             obj_pix_height = dist_to_horizon + dist[upper_region_mask].max()
#             ratios[i] = obj_pix_height / dist_to_horizon
#         else:
#             dist_to_horizon = dist.max()
#             obj_pix_height = dist_to_horizon - dist.min()
#             ratios[i] = obj_pix_height / dist_to_horizon
#     return ratios


def horizon_to_2pts(horizons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bs = horizons.shape[0]
    device = horizons.device
    zeros = torch.zeros((bs,), device=device)
    ones = torch.ones((bs,), device=device)
    return torch.stack((zeros, -horizons[:, 2] / horizons[:, 1]), dim=1), torch.stack(
        (ones, -(horizons[:, 0] + horizons[:, 2]) / horizons[:, 1]), dim=1
    )


# この方法では１つのインスタンスが分かれている時に隙間の部分が考慮されない
# def masks_to_pix_heights(masks: torch.Tensor) -> torch.Tensor:
#     """
#     masks: Size(n_inst, img_h, img_w)
#     """
#     # 画像の行方向に論理和を取ったものの列方向の総和はピクセル高さ
#     return (masks.sum(2) > 0).sum(1)


def masks_to_pix_heights(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: Size(n_inst, img_h, img_w)
    """
    pix_heights = torch.zeros((masks.shape[0],), device=masks.device, dtype=torch.int16)
    for idx, mask in enumerate(masks):
        y, _ = torch.where(mask != 0)
        pix_heights[idx] = torch.max(y) - torch.min(y)
    return pix_heights


def argmax_3d(arr: torch.Tensor) -> torch.Tensor:
    width = arr.shape[-1]
    _, max_idxs = arr.view(arr.shape[0], -1).max(-1)
    return torch.stack([max_idxs // width, max_idxs % width], -1)


def generate_cam_grid(h: int, w: int, invK: torch.Tensor) -> torch.Tensor:
    x_pix, y_pix = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    pix_grid = torch.stack((x_pix, y_pix, torch.ones((h, w))))  # [3,h,w] ([:,x,y]がpixel x, yにおけるhomogeneous vector)
    return (invK[:3, :3] @ pix_grid.reshape(3, -1)).reshape(3, h, w)


def generate_homo_pix_grid(h: int, w: int) -> torch.Tensor:
    x_pix, y_pix = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    return torch.stack((x_pix, y_pix, torch.ones((h, w))))  # [3, h, w]


def depths2cam_pts(batch_depth: torch.Tensor, cam_grid: torch.Tensor) -> torch.Tensor:
    # batch_depth: Size(bs, h, w)
    # cam_grid: Size(3, h, w)
    return cam_grid.permute(1, 2, 0).unsqueeze(0) * batch_depth.unsqueeze(-1)  # [bs, h, w, 3]


def cam_pts2cam_heights(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return A @ normal


def cam_pts2cam_heights_with_normal(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return A @ normal, normal


def cam_pts2normal(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return normal


def cam_pts2cam_height_with_cross_prod(batch_cam_pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # batch_cam_pts: [bs, h, w, 3]
    v0 = batch_cam_pts.roll(-1, dims=2) - batch_cam_pts  # 右横の点を左に移動させる
    v1 = batch_cam_pts.roll((1, -1), dims=(1, 2)) - batch_cam_pts
    v2 = batch_cam_pts.roll(1, dims=1) - batch_cam_pts
    v3 = batch_cam_pts.roll((1, 1), dims=(1, 2)) - batch_cam_pts
    v4 = batch_cam_pts.roll(1, dims=2) - batch_cam_pts
    v5 = batch_cam_pts.roll((-1, 1), dims=(1, 2)) - batch_cam_pts
    v6 = batch_cam_pts.roll(-1, dims=1) - batch_cam_pts
    v7 = batch_cam_pts.roll((-1, -1), dims=(1, 2)) - batch_cam_pts

    normal_sum = torch.zeros_like(batch_cam_pts, device=batch_cam_pts.device)
    vecs = (v0, v1, v1, v2, v3, v4, v5, v6, v7)
    for i in range(8):
        if i + 2 < 8:
            normal_sum += torch.cross(vecs[i], vecs[i + 2], dim=-1)
        else:
            normal_sum += torch.cross(vecs[i], vecs[i + 2 - 8], dim=-1)
    batch_normal = normal_sum / torch.linalg.norm(normal_sum, dim=-1).unsqueeze(-1)  # [bs, h, w, 3]
    batch_cam_height = torch.einsum("ijkl,ijkl->ijk", batch_cam_pts, -batch_normal)  # [bs, h, w]
    return batch_cam_height, batch_normal


def weighted_quantile(arr: torch.Tensor, weights: torch.Tensor, q=0.5):
    # arr: 1D tensor
    # weights: 1D tensor
    non_zeros = weights > 0
    arr = arr[non_zeros]
    weights = weights[non_zeros]
    if arr.shape[0] == 0:
        return torch.nan
    idxs = torch.argsort(arr)
    cum = torch.cumsum(weights[idxs], dim=0)
    q_idx = torch.searchsorted(cum, q * cum[-1])
    if q_idx == idxs.shape[0] - 1:
        return arr[idxs[q_idx]]
    return torch.where(cum[q_idx] / cum[-1] == q, (arr[idxs[q_idx]] + arr[idxs[q_idx + 1]]) / 2, arr[idxs[q_idx]])


def quantile_wo_zeros(arr: torch.Tensor, weights: torch.Tensor, q=0.5):
    # arr: 1D tensor
    # weights: 1D tensor
    non_zero_bools = weights > 0
    arr_wo_zeros = arr[non_zero_bools]
    if arr_wo_zeros.shape[0] == 0:
        return torch.nan
    return arr_wo_zeros.quantile(q)


# def cam_pts2cam_height_with_long_range(batch_cam_pts: torch.Tensor, n_sample: int, batch_road: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     # batch_cam_pts: [bs, h, w, 3]
#     # ランダムに２つのインデックスを各点ごとにサンプリングしてきてその後方向チェック
#     # 片方のベクトルを基準にした時，もう片方のベクトルが180度以内にあれば基準のベクトルが親指に対応．
#     bs, h, w, _ = batch_cam_pts.shape
#     arange_bs = torch.arange(bs)
#     device = batch_cam_pts.device
#     n_pixel = h * w
#     # n_sample = int(n_pixel * sample_ratio)
#     # [bs, h, w, n_sample, 2(A,B), 2(x,y)]

#     x_pix, y_pix = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
#     pix_grid = torch.stack((y_pix.to(device), x_pix.to(device)), dim=-1)[None, :, :, None, None, :]  # [1, h, w, 1, 1, 2]
#     y_idxs = torch.multinomial(torch.ones(h, device=device), bs * n_pixel * n_sample * 2, replacement=True).view(bs, h, w, n_sample, 2)
#     x_idxs = torch.multinomial(torch.ones(w, device=device), bs * n_pixel * n_sample * 2, replacement=True).view(bs, h, w, n_sample, 2)
#     group_idxs = torch.stack((y_idxs, x_idxs), dim=-1)  # [bs, h, w, n_sample, 2(A,B), 2(y,x)]

#     # A, Bの並び替え
#     ## argsortのようにgroup_idxsのidxsを作ってgroup_idxs[idxs]とするイメージ.
#     ## [:, :, :, :, (0,1) or (1,0), :] (:の部分はnp.arange(xxx))
#     ## ２次元ベクトルの外積＝determinant
#     group_idxs_idxs = torch.where(
#         torch.det(group_idxs.float() - pix_grid.float()).unsqueeze(-1) > 0,  # [bs, h, w, n_sample, 1]
#         torch.tensor([0, 1], dtype=torch.long, device=device),
#         torch.tensor([1, 0], dtype=torch.long, device=device),
#     )  # [bs, h, w, n_sample, 2]
#     group_idxs = torch.gather(group_idxs, -2, group_idxs_idxs[..., None].expand(-1, -1, -1, -1, -1, 2))

#     # TODO: あとで１つの式にまとめる
#     batch_cam_pts_flatten = batch_cam_pts.view(bs, n_pixel, 3)  # [bs, h*w, 3]
#     group_idxs = group_idxs[..., 0] * w + group_idxs[..., 1]  # [bs, h, w, n_sample, 2]
#     group_idxs = group_idxs.flatten(start_dim=1)  # [bs, h*w*n_sample*2]
#     group_cam_pts = batch_cam_pts_flatten[arange_bs[:, None], group_idxs, :]  # [bs, h*w*n_sample*2, 3]
#     group_cam_pts = group_cam_pts.view(bs, h, w, n_sample, 2, 3)

#     # TODO: 近すぎるsample点を排除する

#     # 共線のサンプルの外積値は0のはず
#     sample_normals = torch.cross(group_cam_pts[..., 0, :], group_cam_pts[..., 1, :], dim=-1)  # [bs, h, w, n_sample, 3]

#     # そもそも道路マスク内にあるかチェック
#     # group_idxs: [bs, (h * w * n_sample * 2), 2]
#     # mask = batch_road[arange_bs.unsqueeze(1), group_idxs[..., 0], group_idxs[..., 1]].view(bs, h, w, n_sample, 2)
#     # batch_road: [bs, h, w]
#     # group_idxs: [bs, h, w, n_sample, 2, 2]
#     # mask = torch.gather(batch_road[..., None, None, None].expand(-1, -1, -1, n_sample, 2, 2), dim=)

#     batch_road_flatten = batch_road.view(bs, n_pixel)
#     mask = batch_road_flatten[arange_bs[:, None], group_idxs].view(bs, h, w, n_sample, 2)
#     sample_mask = torch.all(mask, dim=-1) * (torch.norm(sample_normals.detach(), dim=-1) != 0.0)  # [bs, h, w, n_sample]
#     batch_normal = (sample_normals * sample_mask.unsqueeze(-1)).sum(3)  # [bs, h, w, 3]
#     batch_normal /= torch.linalg.norm(batch_normal, dim=-1).unsqueeze(-1)  # normの計算時にはmask外のピクセルで計算されたnormalも含まれている
#     batch_cam_height = torch.einsum("ijkl,ijkl->ijk", batch_cam_pts, -batch_normal)
#     return batch_cam_height, batch_normal


# def cam_pts2cam_heights(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
#     # masked_cam_pts: [?, 3]
#     road_pts = cam_pts[road_mask == 1]
#     ones = torch.ones((road_pts.shape[0], 1), dtype=torch.float32, device=road_pts.device)
#     normal = torch.linalg.pinv(road_pts) @ ones
#     # FIXME: LinalgPinvBackward0. No forward pass information availableとなる．
#     # 上のコメントアウトしたコードなら動くが，その場合pinv部分の勾配が伝播しておらず後ろの @ A_T部分の勾配だけ伝わっている可能性がある
#     # TODO: ↑これを確認  torch.autograd.gradcheck(lambda a, b: torch.linalg.pinv(a @ b.t()), [x, y])
#     normal = normal / torch.linalg.norm(normal)
#     return road_pts @ normal


def erode(batch_segms: torch.Tensor, kernel_size: int) -> torch.Tensor:
    eroded_segms = -F.max_pool2d(
        -batch_segms.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=[kernel_size // 2, kernel_size // 2],
    ).to(torch.uint8)
    return eroded_segms


# def erode(segms: torch.Tensor, kernel_size: int) -> torch.Tensor:
#     eroded_segms = -F.max_pool2d(-segms.float(), kernel_size=kernel_size, stride=1, padding=[kernel_size // 2, kernel_size // 2]).to(torch.uint8)
#     return eroded_segms


def readlines(filename):
    """Read all the lines in a text file and return as a list"""
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]"""
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it"""
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
            "a964b8356e08a02d009609d9e3928f7c",
        ),
        "stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
            "3dfb76bcff0786e4ec07ac00f658dd07",
        ),
        "mono+stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
            "c024d69012485ed05d7eaa9617a96b81",
        ),
        "mono_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
            "9c2f071e35027c895a4728358ffc913a",
        ),
        "stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
            "41ec2de112905f85541ac33a854742d1",
        ),
        "mono+stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
            "46c3b824f541d143a45c37df65fbab0a",
        ),
        "mono_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
            "0ab0766efdfeea89a0d9ea8ba90e1e63",
        ),
        "stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
            "afc2f2126d70cf3fdf26b550898b501a",
        ),
        "mono+stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
            "cdc5fc9b23513c07d5b19235d9ef08f7",
        ),
    }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, "rb") as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):
        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", "r") as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
