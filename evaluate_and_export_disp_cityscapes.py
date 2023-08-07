import warnings

warnings.filterwarnings("ignore")


import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import networks
from eval_utils import batch_post_process_disparity, compute_errors, search_last_epoch
from layers import disp_to_depth
from options import MonodepthOptions

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def evaluate_and_export_disp(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    ORIG_WIDTH = 2048
    ORIG_HEIGHT = 1024

    log_path = (
        Path(opt.root_log_dir)
        / f"{opt.width}x{opt.height}"
        / f"{opt.model_name}{'_' if opt.model_name and opt.ckpt_timestamp else ''}{opt.ckpt_timestamp}"
    )
    save_dir: Path = log_path / "result"
    print(f"All results are saved at {save_dir}")
    models_dir = log_path / "models"
    opt.epoch_for_eval = search_last_epoch(models_dir) if opt.epoch_for_eval is None else opt.epoch_for_eval
    weights_dir = log_path / "models" / f"weights_{opt.epoch_for_eval}"

    if not opt.enable_loading_disp_to_eval:
        save_dir.mkdir(parents=False, exist_ok=True)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

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

        dataset = datasets.CityscapesDataset(
            data_path=Path(opt.data_path),
            height=feed_height,
            width=feed_width,
            orig_height=ORIG_HEIGHT,
            orig_width=ORIG_WIDTH,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pred_disps = []

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for input_color in tqdm(dataloader, dynamic_ncols=True):
                input_color = input_color.to(device)

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

        disp_output_path = save_dir / f"disps_cityscapes_{opt.epoch_for_eval}.npy"
        print("-> Saving predicted disps to", disp_output_path)
        np.save(disp_output_path, pred_disps)

    else:
        disp_output_path = save_dir / f"disps_cityscapes_{opt.epoch_for_eval}.npy"
        if not Path(disp_output_path).exists():
            raise FileNotFoundError(f"{disp_output_path} does not exists. Please remove --enable_loading_disp_to_eval option")
        print(f"\n-> Loading predictions from {disp_output_path}")
        pred_disps = np.load(disp_output_path)

    gt_path: Path = Path(splits_dir) / "gt_depths_cityscapes.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)["data"]

    print("-> Evaluating")

    errors = []
    errors_scaling = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        gt_depth[gt_depth > 0] = (gt_depth[gt_depth > 0] - 1) / 256
        gt_depth[gt_depth > 0] = (0.209313 * 2262.52) / gt_depth[gt_depth > 0]
        gt_depth[gt_depth > MAX_DEPTH] = 0
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.05 * gt_height, 0.80 * gt_height, 0.05 * gt_width, 0.99 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
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

    print("|             ", end="")
    print("| " + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("| w/o scaling ", end="")
    print(("|{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "|")
    print("| w/  scaling ", end="")
    print(("|{: 8.3f}  " * 8).format(*mean_errors_scaling.tolist()) + "|")

    output_filename = f"result_{opt.epoch_for_eval}.txt"
    with open(save_dir / output_filename, "w") as f:
        f.write(f"Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}\n\n")
        f.write("\n| " + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        f.write("\n")
        f.write(("|{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "|")
        f.write("\n")
        f.write(("|{: 8.3f}  " * 8).format(*mean_errors_scaling.tolist()) + "|")
        f.write("\n")

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate_and_export_disp(options.parse())
