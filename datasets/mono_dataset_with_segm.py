# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import random
from functools import partial

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms
from torchvision.transforms import functional as F


def load_pil(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def color_aug_func(img, aug_params):
    for fn_idx in aug_params[0]:
        factor = aug_params[fn_idx + 1]
        match fn_idx:
            case 0:
                img = F.adjust_brightness(img, factor)
            case 1:
                img = F.adjust_contrast(img, factor)
            case 2:
                img = F.adjust_saturation(img, factor)
            case 3:
                img = F.adjust_hue(img, factor)
    return img


def load_segms_labels_as_tensor(path: str) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    segms = torch.from_numpy(npz["segms"].astype(np.uint8))
    labels = torch.from_numpy(npz["labels"].astype(np.int8))
    return segms, labels


class MonoDatasetWithSegm(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
        segm_ext
    """

    def __init__(
        self,
        data_path,
        filenames,
        height,
        width,
        adj_frame_idxs,
        num_scales,
        is_train=False,
        img_ext=".jpg",
        segm_ext=".npz",
        segm_dirname="modified_segms_labels_person_car",
    ):
        super().__init__()

        self.data_path = data_path  # ~/workspace/monodepth2/kitti_data
        self.filenames = filenames  # ['2011_10_03/2011_10_03_drive_0034_sync 1757 r', '2011_09_26/2011_09_26_drive_0061_sync 635 l',...]
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.segm_dirname = segm_dirname  # used for get_segm_path() in KITTIRAWDatasetWithSegm
        # self.interp = Image.ANTIALIAS

        self.adj_frame_idxs = adj_frame_idxs  # default: [0, -1, 1]

        self.is_train = is_train
        self.img_ext = img_ext
        self.segm_ext = segm_ext

        self.pil_loader = load_pil
        self.tensor_segms_labels_loader = load_segms_labels_as_tensor
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        img_interp = transforms.InterpolationMode.LANCZOS
        self.resize_func_dict = {}
        self.segm_resize_func_dict = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize_func_dict[i] = transforms.Resize((self.height // s, self.width // s), interpolation=img_interp)
            # self.segm_resize_func_dict[i] = transforms.Resize(
            #     (self.height // s, self.width // s), interpolation=segm_interp
            # )
        self.load_depth = self.check_depth()

    def preprocess(self, input_dict, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # if not using list(), "dict changed size during iteration" error is thrown.
        for key in list(input_dict):
            if key[0] == "color":
                name, f_idx, _ = key
                for scale in range(self.num_scales):
                    # i-1をresize funcへの入力とすることで徐々にresizeしていってる．
                    # TODO: このやり方では縮小しすぎてしまうのでは？論文読んで確かめる
                    input_dict[(name, f_idx, scale)] = self.resize_func_dict[scale](input_dict[(name, f_idx, scale - 1)])
            # elif key[0] == "segms":
            #     name, _ = key
            #     for scale in range(self.num_scales):
            #         # resizeによりsegmが消滅したときの処理を追加
            #         # segmはスケールする必要がない（手法的に，小さくresizeした画像をinputして出てきた深度出力をupscaleするから）
            #         # もしmonodepth1を使う(v1_multiscale)なら，segmもresizeする必要が出てくる
            #         input_dict[(name, scale)] = self.segm_resize_func_dict[scale](input_dict[(name, scale - 1)])

        for key in list(input_dict):
            f = input_dict[key]
            if key[0] == "color":
                name, f_idx, scale = key
                input_dict[(name, f_idx, scale)] = self.to_tensor(f)
                input_dict[(name + "_aug", f_idx, scale)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <adj_frame_idx>, <scale>)          for raw colour images,
            ("color_aug", <adj_frame_idx>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)             for camera intrinsics,
            "stereo_T"                                   for camera extrinsics, and
            "depth_gt"                                   for ground truth depth maps.

        <adj_frame_idx> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        input_dict = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[idx].split()
        scene_name = line[0]

        if len(line) == 3:
            frame_idx = int(line[1])
        else:
            frame_idx = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.adj_frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                input_dict[("color", i, -1)] = self.get_color(scene_name, frame_idx, other_side, do_flip)
            else:
                input_dict[("color", i, -1)] = self.get_color(scene_name, frame_idx + i, side, do_flip)
        # segms and labels doesn't need adjacent frames
        # if self.is_train:
        segms, labels = self.get_segms_labels_tensor(scene_name, frame_idx, side, do_flip)
        input_dict["segms"] = segms
        input_dict["labels"] = labels

        if do_color_aug:
            aug_params = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = partial(color_aug_func, aug_params=aug_params)
        else:
            color_aug = lambda x: x
        self.preprocess(input_dict, color_aug)

        for i in self.adj_frame_idxs:
            del input_dict[("color", i, -1)]
            del input_dict[("color_aug", i, -1)]
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2**scale)
            K[1, :] *= self.height // (2**scale)

            inv_K = np.linalg.pinv(K)

            input_dict[("K", scale)] = torch.from_numpy(K)
            input_dict[("inv_K", scale)] = torch.from_numpy(inv_K)

        if self.load_depth:
            depth_gt = self.get_depth(scene_name, frame_idx, side, do_flip)
            input_dict["depth_gt"] = np.expand_dims(depth_gt, 0)
            input_dict["depth_gt"] = torch.from_numpy(input_dict["depth_gt"].astype(np.float32))

        if "s" in self.adj_frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            input_dict["stereo_T"] = torch.from_numpy(stereo_T)
        return input_dict

    def get_segms_labels_tensor(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError

    def get_color(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_idx, side, do_flip):
        raise NotImplementedError
