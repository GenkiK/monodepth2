import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kitti_horizon.kitti_horizon_raw import KITTIHorizonRaw

BICYCLE_LABEL = 1
MOTORCYCLE_LABEL = 3
INVALID_LABELS = {BICYCLE_LABEL, MOTORCYCLE_LABEL}

gamma = 2.0  # γ値を指定
img2gamma = np.zeros((256, 1), dtype=np.uint8)  # ガンマ変換初期値

for i in range(256):
    img2gamma[i, 0] = 255 * (i / 255) ** (1.0 / gamma)


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


def cam_pts2normal(cam_pts: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    # cam_pts: [h,w,3]
    A = cam_pts[road_mask == 1]  # [?, 3]
    b = -np.ones((A.shape[0], 1), dtype=np.float64)
    A_T = A.T
    normal = np.linalg.pinv(A_T @ A) @ A_T @ b  # [3, 1]
    normal /= np.linalg.norm(normal)
    normal = -normal if normal[1] > 0 else normal
    return normal


def calc_rot_mat(a, b):
    # https://tinyurl.com/2dj8otg5
    a = a.squeeze(-1)
    b = b.squeeze(-1)
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    a2 = np.cross(a, c)
    b2 = np.cross(b, c)
    A = np.vstack([a2, c, a])
    B = np.vstack([b2, c, b])
    return B @ np.linalg.pinv(A)


def horizon2edge_pos(horizon: np.ndarray, h: int, w: int) -> np.ndarray:
    a, b, c = horizon
    xs = np.array((5, w - 5))
    ys = -c / b - a / b * xs
    return xs, ys


def pos2horizon(x1, y1, x2, y2) -> np.ndarray:
    if y1 == y2:
        return np.array((0, 1, -y1))
    elif x1 == x2:
        return np.array((1, 0, -x1))
    else:
        inv = np.linalg.pinv(np.array([[x1, y1], [x2, y2]]))
        return (*(inv @ np.array((-1, -1))), 1)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=320, help="input image height")
    parser.add_argument("--img_width", type=int, default=1024, help="input image width")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    h, w = args.img_height, args.img_width

    camera_numbers = ("2",)
    # camera_numbers = ("2", "3") # FIXME: uncomment this line
    K = np.array([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]], dtype=np.float32)
    K[0, :] *= w
    K[1, :] *= h
    invK = np.linalg.pinv(K)
    cam_grid = generate_cam_grid(h, w, invK)
    resolution = f"{w}x{h}"

    # TODO: erase humpback
    RAW_ROOT_DIR = Path("/home/gkinoshita/humpback/dataset/packnet-kitti-raw/KITTI_raw")
    ROOT_DIR = Path("/home/gkinoshita/humpback/workspace/monodepth2/kitti_data/")

    dataset = KITTIHorizonRaw(dataset_path=RAW_ROOT_DIR, resize_height=h, resize_width=w)
    for date_dir in ROOT_DIR.iterdir():
        if not date_dir.is_dir():
            continue
        for drive_dir in sorted(date_dir.glob("*sync")):
            drive_id = drive_dir.name[-9:-5]
            drive = dataset.get_drive(date_dir.name, drive_id)

            for cam_number in camera_numbers:
                data_dir_tpl = str(drive_dir / ("{}_" + resolution + "_0" + cam_number))
                road_dir = Path(data_dir_tpl.format("road_segm"))
                disp_dir = Path(data_dir_tpl.format("disp"))
                img_dir = Path(data_dir_tpl.format("image"))
                print(img_dir)

                cam_idx = 0 if cam_number == "2" else 1

                for road_idx, road_path in enumerate(sorted(road_dir.glob("*npy"))[:5]):

                    data = dataset.process_single_image(drive, drive.get_rgb(road_idx)[cam_idx], road_idx, int(cam_number))
                    processed_img = data["image"].transpose(1, 2, 0)
                    hp1 = data["horizon_p1"]
                    hp2 = data["horizon_p2"]

                    road_mask = np.load(road_path).astype(np.uint8)
                    road_mask_cut = road_mask.copy()
                    road_mask_cut[: 2 * h // 3, :] = 0
                    if road_mask.sum() == 0:
                        continue

                    disp = np.load(disp_dir / str(road_path.name).replace(".npy", "_disp.npy")).reshape(h, w)
                    depth = 1 / disp

                    cam_pts = depth2cam_pts(depth, cam_grid)
                    normal = cam_pts2normal(cam_pts, road_mask)
                    pred_horizon = invK.T @ normal
                    xs, ys = horizon2edge_pos(pred_horizon, h=h, w=w)

                    normal_cut = cam_pts2normal(cam_pts, road_mask_cut)
                    pred_horizon_cut = invK.T @ normal_cut
                    xs_cut, ys_cut = horizon2edge_pos(pred_horizon_cut, h=h, w=w)
                    print(normal.reshape(3), normal_cut.reshape(3))
                    plt.figure(figsize=(30, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(processed_img)
                    # plt.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], "r-", lw=3, label="GT")
                    xs_gt, ys_gt = horizon2edge_pos(pos2horizon(hp1[0], hp1[1], hp2[0], hp2[1]), h, w)
                    plt.plot(xs_gt, ys_gt, "r-", lw=3, label="GT")
                    # plt.plot(xs, ys, "w-", lw=2)
                    plt.plot(xs_cut, ys_cut, "w-", lw=1, label="pred")
                    # plt.plot(xs_cut, ys_cut, "-", c="dodgerblue", lw=1)
                    plt.legend()
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    road_mask_cut = road_mask_cut.astype(np.float16)
                    road_mask_cut[road_mask_cut == 0] = 0.1
                    plt.imshow(processed_img * np.tile(road_mask_cut, (3, 1, 1)).transpose(1, 2, 0))
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()
                    plt.close()
