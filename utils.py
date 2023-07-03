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
    device = batch_segms.device
    total_n_inst = batch_n_insts.sum()
    ratios = torch.zeros((total_n_inst,), dtype=torch.float32, device=device)
    batch_A, batch_B = horizon_to_2pts(horizons)
    batch_AB = batch_B - batch_A
    norm_AB = torch.norm(batch_AB, dim=1)  # [bs,]
    batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=device)), dim=1)  # add zeros for torch.cross
    dim_y_eraser = torch.tensor([1, 0, 1], dtype=torch.uint8, device=device)[:, None]
    homo_pix_grid = homo_pix_grid.to(device)
    inst_idx_in_batch = 0
    ratio_idxs = torch.where(batch_road_appear_bools.repeat_interleave(batch_n_insts, dim=0))[0]
    for batch_idx in range(horizons.shape[0]):
        for inst_idx in range(batch_n_insts[batch_road_appear_bools][batch_idx]):
            homo_valid_pos = homo_pix_grid[:, batch_segms[batch_idx, inst_idx] == 1]  # [3, n_valid_pos]
            valid_pos = homo_valid_pos[:2]  # [2, n_valid_pos]
            APs = valid_pos - batch_A[batch_idx, :, None]
            n_pts = APs.shape[1]
            APs = torch.cat((APs, torch.zeros((1, n_pts), device=device)), dim=0)  # add zeros for torch.cross
            cross = torch.linalg.cross(batch_AB[batch_idx][:, None].repeat((1, n_pts)), APs, dim=0)[-1]
            dist = torch.abs(cross) / norm_AB[batch_idx]
            upper_region_mask = valid_pos[1] < (-(horizons[batch_idx] @ (homo_valid_pos * dim_y_eraser)) / horizons[batch_idx, 1])  # [n_valid_pos,]
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


def masks_to_pix_heights(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: Size(n_inst, img_h, img_w)
    """
    # 画像の行方向に論理和を取ったものの列方向の総和はピクセル高さ
    return (masks.sum(2) > 0).sum(1)


def argmax_3d(arr: torch.Tensor) -> torch.Tensor:
    width = arr.shape[-1]
    _, max_idxs = arr.view(arr.shape[0], -1).max(-1)
    return torch.stack([max_idxs // width, max_idxs % width], -1)


# def generate_cam_grid(h: int, w: int, invK: torch.Tensor) -> torch.Tensor:
#     x_pix, y_pix = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
#     pix_grid = torch.stack((x_pix, y_pix, torch.ones((h, w))))  # [3,h,w] ([:,x,y]がpixel x, yにおけるhomogeneous vector)
#     return (invK[:3, :3] @ pix_grid.reshape(3, -1)).reshape(3, h, w)


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


def cam_pts2normal(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return normal


# def cam_pts2cam_heights(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
#     A = cam_pts[road_mask == 1].contiguous()
#     # ones = torch.ones((A.shape[0], 1), dtype=torch.float32, device=A.device)
#     ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
#     A_T = A.T
#     normal = torch.linalg.pinv(A_T @ A) @ A_T @ ones
#     # normal = torch.pinverse(A) @ ones
#     normal = normal / torch.linalg.norm(normal)
#     return A @ normal


# def cam_pts2cam_heights(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
#     # masked_cam_pts: [?, 3]
#     road_pts = cam_pts[road_mask == 1]
#     ones = torch.ones((road_pts.shape[0], 1), dtype=torch.float32, device=road_pts.device)
#     breakpoint()
#     normal = torch.linalg.pinv(road_pts) @ ones
#     # FIXME: LinalgPinvBackward0. No forward pass information availableとなる．
#     # 上のコメントアウトしたコードなら動くが，その場合pinv部分の勾配が伝播しておらず後ろの @ A_T部分の勾配だけ伝わっている可能性がある
#     # TODO: ↑これを確認  torch.autograd.gradcheck(lambda a, b: torch.linalg.pinv(a @ b.t()), [x, y])
#     normal = normal / torch.linalg.norm(normal)
#     return road_pts @ normal


# def cam_pts2cam_heights(masked_cam_pts: torch.Tensor) -> torch.Tensor:
#     # masked_cam_pts: [?, 3]
#     ones = torch.ones((masked_cam_pts.shape[0], 1), dtype=torch.float32, device=masked_cam_pts.device)
#     normal = torch.pinverse(masked_cam_pts) @ ones
#     normal = normal / torch.norm(normal)
#     return masked_cam_pts @ normal


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
