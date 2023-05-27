# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import PIL.Image as pil
import skimage.transform
import torch

from kitti_utils import generate_depth_map

from .mono_dataset_with_segm import MonoDatasetWithSegm


class KITTIDatasetWithSegm(MonoDatasetWithSegm):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_idx = int(line[1])

        velo_filename = os.path.join(self.data_path, scene_name, "velodyne_points/data/{:010d}.bin".format(int(frame_idx)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_idx, side, do_flip):
        color = self.pil_loader(self.get_image_path(folder, frame_idx, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_segms_labels_tensor(self, folder, frame_idx, side, do_flip):
        segms, labels = self.tensor_segms_labels_loader(self.get_segm_path(folder, frame_idx, side))
        if do_flip:
            return torch.flip(segms, dims=(2,)), labels
        return segms, labels

    def get_image_path(self, folder, frame_idx, side):
        raise NotImplementedError

    def get_segm_path(self, folder, frame_idx, side):
        raise NotImplementedError


class KITTIRAWDatasetWithSegm(KITTIDatasetWithSegm):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_idx, side):
        f_str = "{:010d}{}".format(frame_idx, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f"image_{self.width}x{self.height}_0{self.side_map[side]}", f_str)
        return image_path

    def get_segm_path(self, folder, frame_idx, side):
        f_str = "{:010d}{}".format(frame_idx, self.segm_ext)
        segm_path = os.path.join(self.data_path, folder, f"{self.segm_dirname}_{self.width}x{self.height}_0{self.side_map[side]}", f_str)
        return segm_path

    def get_depth(self, folder, frame_idx, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(self.data_path, folder, "velodyne_points/data/{:010d}.bin".format(int(frame_idx)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode="constant")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
