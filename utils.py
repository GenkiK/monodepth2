# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import hashlib
import os
import urllib
import zipfile

import torch


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


def generate_cam_grid(h: int, w: int, invK: torch.Tensor) -> torch.Tensor:
    x_pix, y_pix = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    pix_grid = torch.stack((x_pix, y_pix, torch.ones((h, w))))  # [3,h,w] ([:,x,y]がpixel x, yにおけるhomogeneous vector)
    return (invK[:3, :3] @ pix_grid.reshape(3, -1)).reshape(3, h, w)


def depths2cam_pts(batch_depth: torch.Tensor, cam_grid: torch.Tensor) -> torch.Tensor:
    # batch_depth: Size(bs, h, w)
    # cam_grid: Size(3, h, w)
    return cam_grid.permute(1, 2, 0).unsqueeze(0) * batch_depth.unsqueeze(-1)  # [bs, h, w, 3]


def cam_pts2cam_heights(masked_cam_pts: torch.Tensor) -> torch.Tensor:
    # masked_cam_pts: [?, 3]
    ones = torch.ones((masked_cam_pts.shape[0], 1), dtype=torch.float32, device=masked_cam_pts.device)
    normal = torch.pinverse(masked_cam_pts) @ ones
    normal = normal / torch.norm(normal)
    return masked_cam_pts @ normal


# def cam_pts2cam_height(cam_pts: torch.Tensor, road_masks: torch.Tensor) -> torch.Tensor:
#     # cam_pts: [bs, h, w, 3]
#     # road_masks: [bs, h, w]
#     batch_size = cam_pts.shape[0]
#     cam_heights = torch.zeros(batch_size, device=cam_pts.device)
#     # road_maskで切り抜く面積がフレームによって異なるので，バッチで処理できない
#     # for batch_idx in range(batch_size):
#     #     A = cam_pts[batch_idx][road_masks[batch_idx] == 1]  # [?, 3]
#     #     b = -torch.ones((A.shape[0], 1), dtype=torch.float32, device=A.device)
#     #     A_T = A.T
#     #     normal = torch.linalg.pinv(A_T @ A) @ A_T @ b
#     #     normal = normal / torch.linalg.norm(normal)
#     #     cam_heights[batch_idx] = torch.abs(A @ normal).mean()
#     # return cam_heights
#     for batch_idx in range(batch_size):
#         A = cam_pts[batch_idx][road_masks[batch_idx] == 1]  # [?, 3]
#         ones = torch.ones((A.shape[0], 1), dtype=torch.float32, device=A.device)
#         normal = torch.pinverse(A) @ ones
#         normal = normal / torch.norm(normal)
#         cam_heights[batch_idx] = (A @ normal).mean()
#     return cam_heights


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
