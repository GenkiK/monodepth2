import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path("/home/gkinoshita/workspace/monodepth2/kitti_data/")
BICYCLE_LABEL = 1
MOTORCYCLE_LABEL = 3
INVALID_LABELS = {BICYCLE_LABEL, MOTORCYCLE_LABEL}


def masks_to_pix_heights(masks: np.ndarray) -> np.ndarray:
    # 画像の行方向に論理和を取ったものの列方向の総和はピクセル高さ
    return (masks.sum(2) > 0).sum(1).astype(np.uint16)


def argmax_3d(arr: np.ndarray) -> np.ndarray:
    max_idxs = arr.reshape(arr.shape[0], -1).argmax(1)
    return np.column_stack(np.unravel_index(max_idxs, arr[0].shape))


def erode(segms: np.ndarray, kernel) -> tuple[np.ndarray, np.ndarray]:
    eroded_segms = np.array([cv2.erode(segm, kernel) for segm in segms])
    valid_idxs = np.where(eroded_segms.sum(axis=(1, 2)) > 0)[0]
    return eroded_segms[valid_idxs], valid_idxs


# unused
# def exclude_small_segm(segms: np.ndarray, labels: np.ndarray, th: int) -> tuple[np.ndarray, np.ndarray]:
#     areas = segms.sum(axis=(1, 2))
#     sorted_idxs = np.argsort(-areas)
#     segms = segms[sorted_idxs]
#     is_valid_idxs = areas > th
#     segms = segms[is_valid_idxs]
#     labels = labels[is_valid_idxs]
#     return segms, labels


def valid_segms_labels(segms: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_idxs = [i for i, label in enumerate(labels) if label not in INVALID_LABELS]
    return segms[valid_idxs], labels[valid_idxs]


def generate_cam_grid(h, w, invK: np.ndarray):
    x_pix, y_pix = np.meshgrid(np.arange(w), np.arange(h))
    pix_grid = np.stack([x_pix, y_pix, np.ones([h, w])])  # [3,h,w] ([:,x,y]がpixel x, yにおけるhomogeneous vector)
    return (invK[:3, :3] @ pix_grid.reshape(3, -1)).reshape(3, h, w)


def depth2cam_pts(depth: np.ndarray, cam_grid: np.ndarray) -> np.ndarray:
    return (cam_grid * depth).transpose(1, 2, 0)  # [h,w,3]


def cam_pts2cam_height(cam_pts: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    # cam_pts: [h,w,3]
    A = cam_pts[road_mask == 1]  # [?, 3]
    b = -np.ones((A.shape[0], 1), dtype=np.float64)
    A_T = A.T
    normal = np.linalg.pinv(A_T @ A) @ A_T @ b  # [3, 1]
    normal /= np.linalg.norm(normal)
    cam_height = np.abs(A @ normal).mean()
    return cam_height


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=320, help="input image height")
    parser.add_argument("--img_width", type=int, default=1024, help="input image width")
    parser.add_argument("--kernel_size", type=int, default=5, help="the size of kernel for eroding instance segment")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    h, w = args.img_height, args.img_width
    k_size = args.kernel_size
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    camera_numbers = ("2", "3")
    K = np.array([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]], dtype=np.float32)
    K[0, :] *= w
    K[1, :] *= h
    fy = K[1, 1]
    invK = np.linalg.pinv(K)
    cam_grid = generate_cam_grid(h, w, invK)

    # FIXME: 一旦
    # height_priors = np.loadtxt(ROOT_DIR / "height_priors.txt", dtype=np.float32)
    height_priors = np.array(
        [
            [1.747149944305419922e00, 6.863836944103240967e-02],
            [np.nan, np.nan],
            [1.5260834, 0.01868551],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    # KITTI dataset
    #
    # {'Car': (1.5260834, 0.01868551),
    #  'Pedestrian': (1.7607065, 0.012825643),
    # }
    resolution = f"{w}x{h}"
    with open(ROOT_DIR / f"cam_height_{resolution}.csv", "w") as f:
        writer = csv.writer(f)
        header = ("expectation", "variance", "n_inst")
        writer.writerow(header)
        for date_dir in sorted(ROOT_DIR.iterdir()):
            if not date_dir.is_dir():
                continue
            for scene_dir in sorted(date_dir.glob("*sync")):
                for camera_number in camera_numbers:
                    data_dir_tpl = str(scene_dir / ("{}_" + resolution + "_0" + camera_number))
                    road_dir = Path(data_dir_tpl.format("road_segm"))
                    segms_labels_dir = Path(data_dir_tpl.format("modified_segms_labels_person_car"))
                    disp_dir = Path(data_dir_tpl.format("disp"))
                    img_dir = Path(data_dir_tpl.format("image"))

                    basis_normal = None
                    road_paths = sorted(road_dir.glob("*npy"))
                    n_frame = len(road_paths)
                    cam_heights = np.full((n_frame, 3), np.nan, dtype=np.float64)
                    for frame_idx, road_path in enumerate(tqdm(road_paths)):
                        road_mask = np.load(road_path)
                        if road_mask.sum() == 0:
                            continue
                        segms_labels = np.load(segms_labels_dir / f"{road_path.stem}.npz")
                        segms, labels = valid_segms_labels(segms_labels["segms"], segms_labels["labels"])
                        if segms.shape[0] == 0:
                            continue

                        sorted_idxs = segms.sum(axis=(1, 2)).argsort()
                        segms = segms[sorted_idxs]
                        labels = labels[sorted_idxs]

                        disp = np.load(disp_dir / str(road_path.name).replace(".npy", "_disp.npy")).reshape(h, w)
                        depth = 1 / disp
                        eroded_segms, erode_valid_idxs = erode(segms, erode_kernel)
                        if erode_valid_idxs.shape[0] == 0:
                            continue

                        pix_heights = masks_to_pix_heights(segms)[erode_valid_idxs]
                        height_expects_vars = height_priors[labels[erode_valid_idxs]]
                        depth_expects = height_expects_vars[:, 0] / pix_heights * fy

                        # calc nearest point of each segm
                        cam_pts = depth2cam_pts(depth, cam_grid)
                        masked_cam_pts = cam_pts[None, ...] * eroded_segms[..., None]
                        masked_cam_pts[masked_cam_pts == 0] = 1000
                        nearest_pts = argmax_3d(-np.linalg.norm(masked_cam_pts, axis=3))
                        nearest_depths = depth[nearest_pts[:, 0], nearest_pts[:, 1]]

                        # TODO: meanで計算したらどうなるか？->これでいけるなら損失として設定するのでも希望が見えてくる

                        n_inst = height_expects_vars.shape[0]
                        frame_scale_expect = (depth_expects / nearest_depths).mean()
                        frame_scale_var = (height_expects_vars[:, 1] / (pix_heights * nearest_depths / fy) ** 2).sum() / n_inst**2

                        # scale cam_height after calculating scale-free cam_height
                        unscaled_cam_height = cam_pts2cam_height(cam_pts, road_mask)
                        frame_cam_height_expect = frame_scale_expect * unscaled_cam_height
                        frame_cam_height_var = frame_scale_var * unscaled_cam_height**2
                        cam_heights[frame_idx] = (frame_cam_height_expect, frame_cam_height_var, n_inst)

                        # # calculate cam_height after scaling depth map
                        # scaled_cam_height = cam_pts2cam_height(depth2cam_pts(frame_scale_expect * depth, cam_grid), road_mask)
                        # if frame_cam_height_expect != scaled_cam_height:
                        #     print(frame_cam_height_expect, scaled_cam_height)
                        #     breakpoint()
                    writer.writerows(cam_heights)
