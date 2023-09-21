import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append(".")
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import datasets
import networks
from layers import SSIM, BackprojectDepth, Project3D, disp_to_depth, transformation_from_parameters
from utils import cam_pts2cam_height_with_cross_prod

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


def generate_cam_grid(invK: torch.Tensor) -> torch.Tensor:
    x_pix, y_pix = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    pix_grid = torch.stack((x_pix, y_pix, torch.ones((H, W))))  # [3,h,w] ([:,x,y]がpixel x, yにおけるhomogeneous vector)
    return (invK[:3, :3] @ pix_grid.view(3, -1)).reshape(3, H, W).permute(1, 2, 0)  # [h, w, 3]


def horizon2edge_pos(horizon: np.ndarray, w: int) -> np.ndarray:
    a, b, c = horizon
    xs = np.array((5, w - 5))
    ys = -c / b - a / b * xs
    return xs, ys


def batch_depth2cam_pts(batch_depth: torch.Tensor, cam_grid: torch.Tensor) -> torch.Tensor:
    # batch_depth: [bs, h, w]
    # cam_grid: [h, w, 3]
    # return: [bs, h, w, 3]
    return batch_depth[..., None] * cam_grid[None, ...]


def cam_pts2normal(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return normal


def cam_pts2normal_and_cam_heights(cam_pts: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    A = cam_pts[road_mask == 1]
    pinvs = torch.pinverse(A.detach())
    ones = torch.ones((A.shape[0], 1), device=A.device).type_as(A)
    normal = pinvs @ ones
    normal = normal / torch.linalg.norm(normal)
    return normal, (A @ normal).median()  # HACK: using median() instead of mean()


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


def masks_to_pix_heights(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: Size(n_inst, img_h, img_w)
    """
    # 画像の行方向に論理和を取ったものの列方向の総和はピクセル高さ
    return (masks.sum(2) > 0).sum(1)


def make_flats(
    batch_segms: torch.Tensor,
    batch_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def calc_obj_pix_height_over_dist_to_horizon(
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
    total_n_inst = batch_n_insts.sum()
    ratios = torch.zeros((total_n_inst,), dtype=torch.float32, device=DEVICE)
    batch_A, batch_B = horizon_to_2pts(horizons)
    batch_AB = batch_B - batch_A
    norm_AB = torch.norm(batch_AB, dim=1)  # [bs,]
    batch_AB = torch.cat((batch_AB, torch.zeros((batch_AB.shape[0], 1), device=DEVICE)), dim=1)  # add zeros for torch.cross
    dim_y_eraser = torch.tensor([1, 0, 1], dtype=torch.uint8, device=DEVICE)[:, None]
    homo_pix_grid = homo_pix_grid.to(DEVICE)
    inst_idx_in_batch = 0
    ratio_idxs = torch.where(batch_road_appear_bools.repeat_interleave(batch_n_insts, dim=0))[0]
    for batch_idx in range(horizons.shape[0]):
        for inst_idx in range(batch_n_insts[batch_road_appear_bools][batch_idx]):
            homo_valid_pos = homo_pix_grid[:, batch_segms[batch_idx, inst_idx] == 1]  # [3, n_valid_pos]
            valid_pos = homo_valid_pos[:2]  # [2, n_valid_pos]
            APs = valid_pos - batch_A[batch_idx, :, None]
            n_pts = APs.shape[1]
            APs = torch.cat((APs, torch.zeros((1, n_pts), device=DEVICE)), dim=0)  # add zeros for torch.cross
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


def generate_homo_pix_grid() -> torch.Tensor:
    x_pix, y_pix = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    return torch.stack((x_pix, y_pix, torch.ones((H, W))))  # [3, h, w]


def predict_poses(input_dict):
    """Predict poses between input frames for monocular sequences."""
    output_dict = {}
    pose_inputs = torch.cat([input_dict[("color", i, 0)] for i in ADJ_FRAME_IDXS], 1).cuda()
    pose_inputs = [pose_encoder(pose_inputs)]
    axisangle, translation = pose_decoder(pose_inputs)
    # output_dict[("axisangle", 0, 1)] = axisangle
    # output_dict[("translation", 0, 1)] = translation
    output_dict[("cam_T_cam", 0, 1)] = transformation_from_parameters(axisangle.cpu()[:, 0], translation.cpu()[:, 0])
    return output_dict


backproject_depth_func = BackprojectDepth(BS, H, W)
project_3d_func = Project3D(BS, H, W)
ssim_func = SSIM()


def generate_images_pred(input_dict, output_dict, scaled_depth):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    cam_points = backproject_depth_func(scaled_depth.unsqueeze(1).cpu(), input_dict[("inv_K", 0)])
    T = output_dict[("cam_T_cam", 0, 1)]
    pix_coords = project_3d_func(cam_points, input_dict[("K", 0)], T)
    generated_img = F.grid_sample(
        input_dict[("color", 1, 0)],
        pix_coords,
        padding_mode="border",
        align_corners=True,
    )
    return generated_img


def compute_reprojection_loss(pred, target):
    """
    pred: [bs, h, w, 3]
    target: [bs, h, w, 3]
    """
    mean_abs_diff = torch.abs(target - pred).mean(1, True)
    ssim_loss = ssim_func(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * mean_abs_diff
    return reprojection_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=str, default="last")
    parser.add_argument("--th", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    args = get_args()
    if args.epoch.isdigit():
        args.epoch = int(args.epoch)
    epoch = args.epoch

    max_show_img = None
    homo_pix_grid = generate_homo_pix_grid()
    outlier_relative_error_th = args.th

    K = torch.tensor([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]], dtype=torch.float32)
    K[0, :] *= W
    K[1, :] *= H
    fy = K[1, 1]
    invK = torch.pinverse(K)
    cam_grid = generate_cam_grid(invK).cuda()  # [h, w, 3]
    invKs = invK.cuda().repeat((BS, 1, 1))

    # model_ckpt = "vadepth_06-30-12:35"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq5_remove_outliers0.2_abs_mean_after_abs_gradual_hybrid0.01_1_07-01-18:17"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_wo_1st_update_update_freq5_remove_outliers0.3_abs_mean_after_abs_gradual_hybrid0.01_1_07-01-16:03"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_abs_mean_after_abs_gradual_hybrid0.01_1_07-03-12:00"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_disable_road_masking_abs_mean_after_abs_gradual_hybrid0.01_1_07-07-10:31"
    # model_ckpt = "person_car_annot_height_mean_after_abs_gradual_rough1_07-10-12:35"
    # model_ckpt = "person_car_annot_height_mean_after_abs_rough1_06-19-19:01"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_abs_mean_after_abs_gradual_hybrid0.01_1_07-06-17:57"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_8neighbors_normal_loss_abs_mean_after_abs_gradual_hybrid0.01_1_07-18-15:17"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_disable_road_masking_8neighbors_gradual_normal_loss_abs_mean_after_abs_gradual_hybrid0.01_1_07-18-16:30"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1_07-18-15:11"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq5_remove_outliers0.2_disable_road_masking_8neighbors_gradual_normal_loss_abs_mean_after_abs_gradual_hybrid0.01_1_07-18-16:30"
    # model_ckpt = "person_car_annot_height_bs8_wo_scale_dividing_quartile_depth_median_cam_height_median_scale_wo_1st_update_update_freq3_remove_outliers0.2_disable_road_masking_8neighbors_abs_mean_after_abs_gradual_hybrid0.01_1_07-18-15:12"
    model_ckpt = "person_car_annot_height-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-disable_road_masking-8neighbors-from_ground-abs-mean_after_abs-gradual_hybrid0.01_1_08-29-21:20"
    # model_ckpt = "person_car_annot_height-bs8-wo_scale_dividing-median_cam_height-wo_1st_update-update_freq3-remove_outliers0.2-disable_road_masking-8neighbors-abs-mean_after_abs-gradual_hybrid0.01_1_08-29-15:32/"

    root_log_dir = WORK_DIR / "new_logs/1024x320"
    log_path = root_log_dir / model_ckpt
    propose_epochs(log_path)

    models_dir = log_path / "models"
    if epoch == "last":
        epoch = search_last_epoch(models_dir)
    print(f"selected {epoch} epoch!")
    weights_dir = models_dir / f"weights_{epoch}"

    encoder_path = weights_dir / "encoder.pth"
    pose_encoder_path = weights_dir / "pose_encoder.pth"
    decoder_path = weights_dir / "depth.pth"
    pose_decoder_path = weights_dir / "pose.pth"
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

    pose_encoder = networks.ResnetEncoder(18, False, 2)
    loaded_dict_pose_enc = torch.load(pose_encoder_path, map_location=DEVICE)
    filtered_dict_pose_enc = {k: v for k, v in loaded_dict_pose_enc.items() if k in pose_encoder.state_dict()}
    pose_encoder.load_state_dict(filtered_dict_pose_enc)
    pose_encoder.to(DEVICE)
    pose_encoder.eval()

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    loaded_dict_pose_dec = torch.load(pose_decoder_path, map_location=DEVICE)
    pose_decoder.load_state_dict(loaded_dict_pose_dec)
    pose_decoder.to(DEVICE)
    pose_decoder.eval()

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
        train_filenames,
        H,
        W,
        [0, 1],
        1,
        is_train=False,
        segm_dirname="modified_segms_labels_person_car_road",
    )
    dataloader = DataLoader(dataset, BS, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    n_shown_img = 0
    n_row = 5
    with torch.no_grad():
        for input_dict in dataloader:
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

            features = encoder(batch_color)
            output_dict = decoder(features)
            output_dict.update(predict_poses(input_dict))

            batch_disp, batch_depth = disp_to_depth(output_dict[("disp", 0)], MIN_DEPTH, MAX_DEPTH)

            batch_disp = batch_disp.cpu()[:, 0].numpy()
            batch_depth = batch_depth.squeeze()
            batch_road = input_dict["road"].cuda()
            batch_pred_color = generate_images_pred(input_dict, output_dict, batch_depth)
            reprojection_loss = compute_reprojection_loss(batch_pred_color, batch_color_cpu)
            identity_reprojection_loss = compute_reprojection_loss(input_dict[("color", 1, 0)], batch_color_cpu)
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape) * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            _, idxs = torch.min(combined, dim=1)
            automask = idxs > identity_reprojection_loss.shape[1] - 1
            batch_pred_color = batch_pred_color.permute(0, 2, 3, 1)

            batch_color = (255 * batch_color).to(torch.uint8).permute((0, 2, 3, 1))

            road_appear_bools = batch_road.sum((1, 2)) > MIN_ROAD_AREA
            bs_wo_no_road = road_appear_bools.sum()
            batch_cam_pts_wo_no_road = batch_depth2cam_pts(batch_depth[road_appear_bools], cam_grid)
            batch_normal = torch.zeros((bs_wo_no_road, 3, 1), device=DEVICE)
            batch_cam_height = torch.zeros((bs_wo_no_road,), device=DEVICE)
            batch_normal_with_pinv = torch.zeros((bs_wo_no_road, 3, 1), device=DEVICE)
            batch_cam_height_with_pinv = torch.zeros((bs_wo_no_road,), device=DEVICE)
            batch_road_wo_no_road = batch_road[road_appear_bools]
            road_img = batch_color * batch_road[..., None]
            bg_regions = road_img.sum(-1)
            red_mask = road_img[bg_regions == 0]
            red_mask[..., 0] = 255
            road_img[bg_regions == 0] = red_mask
            road_img = road_img.cpu().numpy()
            batch_color = batch_color.cpu().numpy()

            for batch_idx in range(bs_wo_no_road):
                batch_normal_with_pinv[batch_idx], batch_cam_height_with_pinv[batch_idx] = cam_pts2normal_and_cam_heights(
                    batch_cam_pts_wo_no_road[batch_idx], batch_road_wo_no_road[batch_idx]
                )
            horizons_with_pinv = (invKs[road_appear_bools].transpose(1, 2) @ batch_normal_with_pinv).squeeze()  # [bs_wo_no_road, 3]

            cam_heights, normals = cam_pts2cam_height_with_cross_prod(batch_cam_pts_wo_no_road)
            # cam_heights, normals = cam_pts2cam_height_with_long_range(batch_cam_pts_wo_no_road, N_SAMPLE, batch_road_wo_no_road)
            batch_road_wo_no_road[:, -1, :] = 0
            batch_road_wo_no_road[:, :, 0] = 0
            batch_road_wo_no_road[:, :, -1] = 0
            batch_nan_masks = ~torch.isnan(cam_heights)  # [bs, h, w]
            for batch_idx in range(bs_wo_no_road):
                batch_cam_height[batch_idx] = (cam_heights[batch_idx, (batch_road_wo_no_road[batch_idx] == 1) * batch_nan_masks[batch_idx]]).quantile(
                    0.5
                )
                batch_normal[batch_idx] = (
                    (normals[batch_idx, (batch_road_wo_no_road[batch_idx] == 1) * batch_nan_masks[batch_idx]]).quantile(0.5, dim=0).unsqueeze(-1)
                )

            horizons = (invKs[road_appear_bools].transpose(1, 2) @ batch_normal).squeeze()  # [bs_wo_no_road, 3]

            if n_inst_appear_frames > 0:
                obj_pix_height_over_dist_to_horizon = calc_obj_pix_height_over_dist_to_horizon(
                    homo_pix_grid, batch_segms, horizons, batch_n_insts, road_appear_bools
                )
                approx_heights = obj_pix_height_over_dist_to_horizon * cam_height_expect[0]
                relative_err = (approx_heights - obj_height_expects).abs() / obj_height_expects
                outlier_bools = relative_err > outlier_relative_error_th
                inlier_bools = ~outlier_bools
                batch_n_insts_lst = batch_n_insts.tolist()
                batch_n_outlier_insts = torch.tensor([chunk.sum() for chunk in torch.split(outlier_bools, batch_n_insts_lst)])
                cumsum_n_outlier_insts = batch_n_outlier_insts.cumsum(dim=0, dtype=torch.int16).numpy()
                batch_n_inlier_insts = torch.tensor([chunk.sum() for chunk in torch.split(inlier_bools, batch_n_insts_lst)], device=DEVICE)
                prev_n_inst = 0
                for batch_idx, n_inst in enumerate(batch_n_inlier_insts):
                    img = batch_color[batch_idx]
                    n_inst = n_inst.item()
                    for i in range(n_inst):
                        mask = (segms_flat[inlier_bools][prev_n_inst + i] == 1).cpu().numpy()
                        img[mask] = img[mask] * ALPHA + np.array([int(255 * color) for color in TAB10_CMAP(i)[:3]], dtype=np.uint8) * (1 - ALPHA)
                    batch_color[batch_idx] = img
                    prev_n_inst += n_inst

                outlier_boxes = masks_to_boxes(segms_flat[outlier_bools])
                outlier_rects = make_rects(outlier_boxes.cpu().numpy())

                # calculate scale
                depth_repeat = batch_depth.repeat_interleave(batch_n_inlier_insts, dim=0)
                masked_depth_repeat = depth_repeat * segms_flat[inlier_bools]
                quartile_depths = torch.zeros((masked_depth_repeat.shape[0],), dtype=torch.float32, device=DEVICE)
                for i in range(masked_depth_repeat.shape[0]):
                    quartile_depths[i] = masked_depth_repeat[i][masked_depth_repeat[i] > 0].quantile(q=0.25)
                depth_expects = obj_height_expects[inlier_bools] * fy / obj_pix_heights[inlier_bools]
                split_scale = torch.split(depth_expects / quartile_depths, batch_n_inlier_insts.tolist())
                for lst_elem in split_scale:
                    print(lst_elem.cpu().numpy(), end=", ")
                print("  black percentage", (AREA - automask.sum((1, 2)).cpu().numpy()) / AREA)
                # HACK: using median() instead of mean()
                frame_scales = torch.tensor([chunk.median() for chunk in split_scale], device="cpu")

            fig, axes = plt.subplots(n_row, BS, tight_layout=True, figsize=(BS * 9, n_row * 3))

            horizon_idx = 0
            horizons = horizons.cpu().numpy()
            horizons_with_pinv = horizons_with_pinv.cpu().numpy()
            batch_cam_height = batch_cam_height.cpu().numpy()
            for batch_idx in range(BS):
                row_idx = 0
                if road_appear_bools[batch_idx]:
                    xs_pred, ys_pred = horizon2edge_pos(horizons[horizon_idx], W)
                    xs_pred_with_pinv, ys_pred_with_pinv = horizon2edge_pos(horizons_with_pinv[horizon_idx], W)
                axes[row_idx, batch_idx].imshow(batch_color[batch_idx])
                if road_appear_bools[batch_idx]:
                    axes[row_idx, batch_idx].plot(xs_pred, ys_pred, c="red", label="pred")
                    axes[row_idx, batch_idx].plot(xs_pred_with_pinv, ys_pred_with_pinv, c="dodgerblue", label="pred with pinv")
                    axes[row_idx, batch_idx].legend()
                    axes[row_idx, batch_idx].set_title(
                        f"in:{batch_n_inlier_insts[batch_idx]}, out:{batch_n_outlier_insts[batch_idx]}, {batch_cam_height[horizon_idx]:.2f} -> {batch_cam_height[horizon_idx] * frame_scales[batch_idx]:.2f}"
                    )

                    horizon_idx += 1
                start_idx = cumsum_n_outlier_insts[batch_idx - 1] if batch_idx > 0 else 0
                if n_inst_appear_frames > 0:
                    for inst_idx in range(start_idx, cumsum_n_outlier_insts[batch_idx]):
                        axes[row_idx, batch_idx].add_patch(outlier_rects[inst_idx])
                axes[row_idx, batch_idx].axis("off")
                row_idx += 1
                axes[row_idx, batch_idx].imshow((batch_pred_color[batch_idx] * 255).to(torch.uint8))
                axes[row_idx, batch_idx].axis("off")
                row_idx += 1
                axes[row_idx, batch_idx].imshow(automask[batch_idx], cmap="gray")
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
