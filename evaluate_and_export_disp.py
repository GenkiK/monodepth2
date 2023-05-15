# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import networks
from layers import disp_to_depth
from options import MonodepthOptions

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abse = np.mean(np.abs(gt - pred))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abse, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# TODO: post_processはデフォルトで必要なのか調べる
def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1"""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate_and_export_disp(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    log_path = (
        Path(opt.root_log_dir)
        / f"{opt.width}x{opt.height}"
        / f"{opt.model_name}{'_' if opt.model_name and opt.ckpt_timestamp else ''}{opt.ckpt_timestamp}"
    )
    save_dir: Path = log_path / "result"
    print(f"All results are saved at {save_dir}")

    if opt.disp_filename_to_eval is None:
        save_dir.mkdir(parents=False, exist_ok=True)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if opt.epoch_for_eval is None:
            weights_dir = log_path / "models" / "best_weights"
        else:
            weights_dir = log_path / "models" / f"weights_{opt.epoch_for_eval}"

        with open(os.path.join(splits_dir, opt.eval_split, "test_files.txt")) as f:
            filenames = f.readlines()

        print("-> Loading model from ", weights_dir)
        encoder_path = weights_dir / "encoder.pth"
        depth_decoder_path = weights_dir / "depth.pth"

        # LOADING PRETRAINED MODEL
        print("Loading pretrained encoder")
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc["height"]
        feed_width = loaded_dict_enc["width"]
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        print("Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, feed_height, feed_width, [0], 4, is_train=False)
        dataloader = DataLoader(
            dataset,
            16,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pred_disps = []

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for data in tqdm(dataloader, dynamic_ncols=True):
                input_color = data[("color", 0, 0)].to(device)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output_dict = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output_dict[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps)

        disp_output_path = save_dir / f"disps_{opt.eval_split}_split_{'best' if opt.epoch_for_eval is None else opt.epoch_for_eval}.npy"
        print("-> Saving predicted disps to", disp_output_path)
        np.save(disp_output_path, pred_disps)

    else:
        disp_output_path = save_dir / opt.disp_filename_to_eval
        if not Path(disp_output_path).exists():
            raise FileNotFoundError(f"{disp_output_path} does not exists. Please remove --disp_filename_to_eval option")
        print(f"\n-> Loading predictions from {disp_output_path}")
        pred_disps = np.load(disp_output_path)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(splits_dir / "benchmark" / "eigen_to_benchmark_ids.npy")
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    gt_path: Path = Path(splits_dir) / opt.eval_split / "gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)["data"]

    print("-> Evaluating")

    errors = []
    errors_scaling = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        # if not opt.disable_median_scaling:
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        scaled_pred_depth = pred_depth * ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        scaled_pred_depth[scaled_pred_depth < MIN_DEPTH] = MIN_DEPTH
        scaled_pred_depth[scaled_pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))
        errors_scaling.append(compute_errors(gt_depth, scaled_pred_depth))

    # if not opt.disable_median_scaling:
    ratios = np.array(ratios)
    med = np.median(ratios)
    print(f"Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)
    mean_errors_scaling = np.array(errors_scaling).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print(("&{: 8.3f}  " * 8).format(*mean_errors_scaling.tolist()) + "\\\\")

    output_filename = f"result_{'best_model' if opt.epoch_for_eval is None else opt.epoch_for_eval}.txt"
    with open(save_dir / output_filename, "w") as f:
        f.write(f"Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}\n\n")
        f.write("\n  " + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        f.write("\n")
        f.write(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
        f.write("\n")
        f.write(("&{: 8.3f}  " * 8).format(*mean_errors_scaling.tolist()) + "\\\\")
        f.write("\n")

    print("\n-> Done!")


if __name__ == "__main__":
    # args = parse_args()
    # export_disp(args)
    options = MonodepthOptions()
    evaluate_and_export_disp(options.parse())
