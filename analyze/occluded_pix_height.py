import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append(".")
import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import datasets
import networks
from layers import disp_to_depth
from utils import (
    calc_occluded_obj_pix_height,
    cam_pts2cam_height_with_cross_prod,
    depths2cam_pts,
    generate_cam_grid,
    generate_homo_pix_grid,
    masks_to_pix_heights,
)

DEVICE = torch.device("cuda")
ALPHA = 0.5
WORK_DIR = Path("/home/gkinoshita/humpback/workspace/monodepth2")
DATA_DIR = WORK_DIR / "kitti_data"
ADJ_FRAME_IDXS = (0, 1)
BS = 3
H, W = 320, 1024
AREA = H * W
MIN_ROAD_AREA = int(AREA * 0.01)
N_SAMPLE = 10

SPECTRAL_CMAP_NAME = "Spectral"
TAB10_CMAP = plt.get_cmap("tab10")
Set2_CMAP = plt.get_cmap("Set2")
MIN_DEPTH = 0.1
MAX_DEPTH = 100.0
height_priors = torch.tensor(
    [
        [1.747149944305419922e00, 6.863836944103240967e-02],
        [np.nan, np.nan],
        [1.5260834, 0.01868551],
    ],
    dtype=torch.float32,
).cuda()
LARGE_VALUE = 100
segms_labels_str_set = ("segms", "labels")

gamma = 2.0  # ガンマ値
img2gamma = np.zeros((256, 1), dtype=np.uint8)  # ガンマ変換初期値
for i in range(256):
    img2gamma[i, 0] = 255 * (i / 255) ** (1.0 / gamma)


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


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    n_inst = masks.shape[0]
    bboxes = torch.zeros((n_inst, 4), dtype=torch.int16)
    for idx, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        bboxes[idx, 0] = (x).min()
        bboxes[idx, 1] = (y).min()
        bboxes[idx, 2] = (x).max()
        bboxes[idx, 3] = (y).max()
    return bboxes


def make_rects(boxes: np.ndarray) -> list[patches.Rectangle]:
    rects = []
    for idx, box in enumerate(boxes):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=TAB10_CMAP(idx)[:3],
            facecolor="none",
            lw=2.5,
        )
        rects.append(rect)
    return rects


def horizon2edge_pos(horizon: np.ndarray, w: int) -> np.ndarray:
    a, b, c = horizon
    xs = np.array((5, w - 5))
    ys = -c / b - a / b * xs
    return xs, ys


def propose_epochs(log_path: Path) -> None:
    print("epoch options: ", end="")
    for p in (log_path / "models").glob("weights_*"):
        print(p.name.replace("weights_", ""), end=", ")
    print("")


def search_last_epoch(models_dir: Path) -> int:
    last_epoch = -1
    for weights_dir in models_dir.glob("weights_*"):
        epoch = int(weights_dir.name[8:])
        if epoch > last_epoch:
            last_epoch = epoch
    return last_epoch


def make_flats(
    batch_segms: torch.Tensor,
    batch_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segms_flat = batch_segms.view(-1, H, W)
    non_padded_channels = segms_flat.sum(dim=(1, 2)) > 0
    segms_flat = segms_flat[non_padded_channels]  # exclude padded channels
    labels_flat = batch_labels.view(-1)[non_padded_channels].long()
    obj_pix_heights = masks_to_pix_heights(segms_flat)
    obj_height_expects = height_priors[labels_flat, 0]
    return segms_flat, obj_pix_heights, obj_height_expects


def horizon_to_2pts(horizons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bs = horizons.shape[0]  # do "not" change to BS
    zeros = torch.zeros((bs,), device=DEVICE)
    ones = torch.ones((bs,), device=DEVICE)
    return torch.stack((zeros, -horizons[:, 2] / horizons[:, 1]), dim=1), torch.stack(
        (ones, -(horizons[:, 0] + horizons[:, 2]) / horizons[:, 1]), dim=1
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=str, default="last")
    parser.add_argument("--th", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    args = get_args()
    if args.epoch.isdigit():
        args.epoch = int(args.epoch)
    epoch = args.epoch

    max_show_img = None
    homo_pix_grid = generate_homo_pix_grid(H, W)
    outlier_relative_error_th = args.th

    K = torch.tensor([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]], dtype=torch.float32)
    K[0, :] *= W
    K[1, :] *= H
    fy = K[1, 1]
    invK = torch.pinverse(K)
    cam_grid = generate_cam_grid(H, W, invK).cuda()  # [3, h, w]
    invKs = invK.cuda().repeat((BS, 1, 1))

    model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1_07-18-15:12"

    root_log_dir = WORK_DIR / "new_logs/1024x320"
    log_path = root_log_dir / model_ckpt
    propose_epochs(log_path)

    models_dir = log_path / "models"
    if epoch == "last":
        epoch = search_last_epoch(models_dir)
    print(f"selected {epoch} epoch!")
    weights_dir = models_dir / f"weights_{epoch}"

    encoder_path = weights_dir / "encoder.pth"
    decoder_path = weights_dir / "depth.pth"
    cam_height_path = weights_dir / "cam_height_expect.pkl"

    print("\n loading ", end="")
    encoder = networks.ResnetEncoder(18, False, 1)
    loaded_dict_enc = torch.load(encoder_path, map_location=DEVICE)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(DEVICE)
    encoder.eval()

    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc)
    loaded_dict_dec = torch.load(decoder_path, map_location=DEVICE)
    decoder.load_state_dict(loaded_dict_dec)
    decoder.to(DEVICE)
    decoder.eval()

    if cam_height_path.exists():
        with open(cam_height_path, "rb") as f:
            cam_height_expect = pickle.load(f)
            print(f"\ncam_height is loaded: {cam_height_expect[0]:.3g}")
    else:
        cam_height_expect = [1.7]
        print(f"predefined cam_height: {cam_height_expect[0]:.3g}")

    print("\n-> loaded\n")

    train_filepath = WORK_DIR / "splits/eigen_zhou/train_files.txt"
    with open(train_filepath, "r") as f:
        train_filenames = f.readlines()

    dataset = datasets.KITTIRAWDatasetWithRoad(
        DATA_DIR,
        train_filenames[6 * 3 + 1 :],
        # train_filenames[6 * 3 :],
        H,
        W,
        [0, 1],
        1,
        is_train=False,
        segm_dirname="modified_segms_labels_person_car_road",
    )
    dataloader = DataLoader(dataset, BS, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    n_shown_img = 0
    n_row = 3
    with torch.no_grad():
        for input_dict in dataloader:
            print(input_dict["filename"])
            if max_show_img is not None and n_shown_img > max_show_img:
                break
            batch_color_cpu = input_dict[("color", 0, 0)]
            batch_color = batch_color_cpu.cuda()
            batch_n_insts = input_dict["n_insts"].cuda()
            n_inst_appear_frames = (batch_n_insts > 0).sum()
            if n_inst_appear_frames > 0:
                batch_segms = input_dict["padded_segms"].cuda()
                batch_labels = input_dict["padded_labels"].cuda()
                segms_flat, obj_pix_heights, obj_height_expects = make_flats(batch_segms, batch_labels)

            output_dict = decoder(encoder(batch_color))
            batch_disp, batch_depth = disp_to_depth(output_dict[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
            batch_disp = batch_disp.cpu()[:, 0].numpy()
            batch_depth = batch_depth.squeeze()
            batch_road = input_dict["road"].cuda()

            road_appear_bools = batch_road.sum((1, 2)) > MIN_ROAD_AREA
            bs_wo_no_road = road_appear_bools.sum()
            batch_cam_pts_wo_no_road = depths2cam_pts(batch_depth[road_appear_bools], cam_grid)

            batch_road_normal_wo_no_road = torch.zeros((bs_wo_no_road, 3), device=DEVICE)
            batch_road_cam_height_wo_no_road = torch.zeros((bs_wo_no_road,), device=DEVICE)
            batch_road_wo_no_road = batch_road[road_appear_bools]

            # create road image with the red background
            batch_color = (255 * batch_color).to(torch.uint8).permute((0, 2, 3, 1))
            road_img = batch_color * batch_road[..., None]
            bg_regions = road_img.sum(-1)
            red_mask = road_img[bg_regions == 0]
            red_mask[..., 0] = 255
            road_img[bg_regions == 0] = red_mask
            road_img = road_img.cpu().numpy()
            batch_color = batch_color.cpu().numpy()

            # 画像領域各ピクセルのカメラ高さと法線ベクトル
            batch_cam_heights_wo_no_road, batch_normals_wo_no_road = cam_pts2cam_height_with_cross_prod(batch_cam_pts_wo_no_road)
            batch_road_wo_no_road[:, -1, :] = 0
            batch_road_wo_no_road[:, :, 0] = 0
            batch_road_wo_no_road[:, :, -1] = 0
            for batch_idx_wo_no_road in range(bs_wo_no_road):
                batch_road_cam_height_wo_no_road[batch_idx_wo_no_road] = (
                    batch_cam_heights_wo_no_road[batch_idx_wo_no_road, (batch_road_wo_no_road[batch_idx_wo_no_road] == 1)]
                ).quantile(0.5)
                sum_normal = batch_normals_wo_no_road[batch_idx_wo_no_road, (batch_road_wo_no_road[batch_idx_wo_no_road] == 1)].sum(0)
                batch_road_normal_wo_no_road[batch_idx_wo_no_road] = sum_normal / torch.norm(sum_normal)

            horizons_wo_no_road = (
                invKs[road_appear_bools].transpose(1, 2) @ batch_road_normal_wo_no_road.unsqueeze(-1)
            ).squeeze()  # [bs_wo_no_road, 3]

            if n_inst_appear_frames > 0:
                # detect outliers with horizon
                # obj_pix_height_over_dist_to_horizon = calc_obj_pix_height_over_dist_to_horizon(
                #     homo_pix_grid, batch_segms, horizons_wo_no_road, batch_n_insts, road_appear_bools
                # )
                # approx_heights = obj_pix_height_over_dist_to_horizon * cam_height_expect[0]
                # relative_err = (approx_heights - obj_height_expects).abs() / obj_height_expects
                # outlier_bools = relative_err > outlier_relative_error_th
                # inlier_bools = ~outlier_bools
                top_pos_wo_no_road_flat, bottom_pos_wo_no_road_flat, color_idx_flat = calc_occluded_obj_pix_height(
                    homo_pix_grid,
                    batch_segms,
                    obj_height_expects,
                    cam_height_expect[0],
                    horizons_wo_no_road,
                    batch_n_insts,
                    road_appear_bools,
                )
                top_pos_wo_no_road_flat = top_pos_wo_no_road_flat.cpu().numpy()
                bottom_pos_wo_no_road_flat = bottom_pos_wo_no_road_flat.cpu().numpy()

                # creating requirements for plotting segms
                # cumsum_n_outlier_insts = batch_n_outlier_insts.cumsum(dim=0, dtype=torch.int16).numpy()
                prev_n_inst = 0
                for batch_idx, n_inst in enumerate(batch_n_insts):
                    img = batch_color[batch_idx]
                    n_inst = n_inst.item()
                    for i in range(n_inst):
                        mask = (segms_flat[prev_n_inst + i] == 1).cpu().numpy()
                        img[mask] = img[mask] * ALPHA + np.array([int(255 * color) for color in TAB10_CMAP(i)[:3]], dtype=np.uint8) * (1 - ALPHA)
                    batch_color[batch_idx] = img
                    prev_n_inst += n_inst
                boxes = masks_to_boxes(segms_flat)
                # rects = make_rects(boxes.cpu().numpy())

            fig, axes = plt.subplots(n_row, BS, tight_layout=True, figsize=(BS * 9, n_row * 3))

            horizon_idx = 0
            prev_n_inst = 0
            horizons_wo_no_road = horizons_wo_no_road.cpu().numpy()
            batch_road_cam_height_wo_no_road = batch_road_cam_height_wo_no_road.cpu().numpy()
            batch_n_insts_wo_no_road = batch_n_insts[road_appear_bools]
            batch_idx_wo_no_road = -1
            for batch_idx, n_inst in enumerate(batch_n_insts):
                n_inst = n_inst.item()
                row_idx = 0
                axes[row_idx, batch_idx].imshow(cv2.LUT(batch_color[batch_idx], img2gamma))
                if road_appear_bools[batch_idx]:
                    batch_idx_wo_no_road += 1
                    xs_pred, ys_pred = horizon2edge_pos(horizons_wo_no_road[horizon_idx], W)
                    axes[row_idx, batch_idx].plot(xs_pred, ys_pred, c="red", label="pred")
                    axes[row_idx, batch_idx].legend()
                    horizon_idx += 1
                    # if n_inst_appear_frames > 0:
                    #     for i in range(n_inst):
                    #         axes[row_idx, batch_idx].add_patch(rects[prev_n_inst + i])
                    #     prev_n_inst += n_inst
                    if batch_n_insts_wo_no_road[batch_idx_wo_no_road] > 0:
                        top = top_pos_wo_no_road_flat[prev_n_inst : prev_n_inst + batch_n_insts_wo_no_road[batch_idx_wo_no_road]]
                        bottom = bottom_pos_wo_no_road_flat[prev_n_inst : prev_n_inst + batch_n_insts_wo_no_road[batch_idx_wo_no_road]]
                        color_idxs = color_idx_flat[prev_n_inst : prev_n_inst + batch_n_insts_wo_no_road[batch_idx_wo_no_road]]
                        for i in range(top.shape[0]):
                            if bottom[i, 0] < H:
                                axes[row_idx, batch_idx].plot((top[i, 1], bottom[i, 1]), (top[i, 0], bottom[i, 0]), c=Set2_CMAP(color_idxs[i])[:3])
                        prev_n_inst += batch_n_insts_wo_no_road[batch_idx_wo_no_road]
                axes[row_idx, batch_idx].axis("off")
                row_idx += 1
                axes[row_idx, batch_idx].imshow(batch_disp[batch_idx], cmap=SPECTRAL_CMAP_NAME)
                axes[row_idx, batch_idx].axis("off")
                row_idx += 1
                axes[row_idx, batch_idx].imshow(road_img[batch_idx])
                axes[row_idx, batch_idx].axis("off")
            plt.show()
            plt.close()
            n_shown_img += 1
