import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# TODO: erase humpback
ROOT_DIR = Path("/home/gkinoshita/workspace/monodepth2/kitti_data/")
BICYCLE_LABEL = 1
MOTORCYCLE_LABEL = 3
INVALID_LABELS = {BICYCLE_LABEL, MOTORCYCLE_LABEL}

# TODO: delete below
gamma = 2.0  # ガンマ値
img2gamma = np.zeros((256, 1), dtype=np.uint8)  # ガンマ変換初期値
for i in range(256):
    img2gamma[i, 0] = 255 * (i / 255) ** (1.0 / gamma)


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
def exclude_small_segm(segms: np.ndarray, labels: np.ndarray, th: int) -> tuple[np.ndarray, np.ndarray]:
    areas = segms.sum(axis=(1, 2))
    sorted_idxs = np.argsort(-areas)
    segms = segms[sorted_idxs]
    is_valid_idxs = areas > th
    segms = segms[is_valid_idxs]
    labels = labels[is_valid_idxs]
    return segms, labels


def valid_segms_labels(segms: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_idxs = [i for i, label in enumerate(labels) if label not in INVALID_LABELS]
    return segms[valid_idxs], labels[valid_idxs]


def search_nearest_depths(eroded_segms: np.ndarray, depth: np.ndarray, disp: np.ndarray) -> np.ndarray:
    # eroded_segms = np.array([cv2.erode(segm, kernel) for segm in segms])
    nearest_pts = argmax_3d(disp * eroded_segms)
    return depth[nearest_pts[:, 0], nearest_pts[:, 1]]


def search_nearest_pts(eroded_segms: np.ndarray, disp: np.ndarray) -> np.ndarray:
    # eroded_segms = np.array([cv2.erode(segm, kernel) for segm in segms])
    return argmax_3d(disp * eroded_segms)


def generate_cam_grid(h, w, invK: np.ndarray):
    x_pix, y_pix = np.meshgrid(np.arange(w), np.arange(h))
    pix_grid = np.stack([x_pix, y_pix, np.ones([h, w])])  # [3,h,w] ([:,x,y]がpixel x, yにおけるhomogeneous vector)
    return (invK[:3, :3] @ pix_grid.reshape(3, -1)).reshape(3, h, w)


def depth2cam_pts(depth: np.ndarray, cam_grid: np.ndarray) -> np.ndarray:
    return (cam_grid * depth).transpose(1, 2, 0)  # [h,w,3]


def cam_pts2cam_height(cam_pts: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    # cam_pts: [h,w,3]
    A = cam_pts[road_mask == 1]  # [?, 3]
    ones = -np.ones((A.shape[0], 1), dtype=np.float64)
    normal = np.linalg.pinv(A) @ ones  # [3, 1]
    normal /= np.linalg.norm(normal)
    cam_height = (A @ normal).mean()
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


# maybe wrong and unused
def calc_rot_mat_before(frm, to):
    # shorturl.at/jsX68
    frm = frm.squeeze(-1)
    to = to.squeeze(-1)
    if np.array_equal(frm, to):
        print("parallel")
        return np.identity(3)
    if np.array_equal(frm, -to):
        print("parallel")
        return -np.identity(frm, to)
    s = frm + to
    return 2 * np.outer(s, s) / np.dot(s, s) - np.identity(3)


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


def plot_normal_map(normal: np.ndarray) -> None:
    plt.figure(figsize=(15, 5))
    plt.imshow((normal * np.array((1, -1, -1)) + 1) / 2)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


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

    # camera_numbers = ("2", "3")
    camera_numbers = ("2",)
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
    #  'Van': (2.2065923, 0.10561367),
    #  'Truck': (3.2517095, 0.20146538),
    #  'Pedestrian': (1.7607065, 0.012825643),
    #  'Person_sitting': (1.2749549, 0.012091665),
    #  'Cyclist': (1.7372034, 0.008986217),
    #  'Tram': (3.5289235, 0.030839745)
    # }

    # iou_thを一律で0.3
    # person: [1.62759197 0.05733813]
    # bicycle: [1.58420002 0.04247103]
    # car: [1.58751416 0.07041512]
    # motorcycle: [1.56388891 0.0209571 ]
    # airplane: [nan nan]
    # bus: [1.90837204 0.28415084]
    # train: [2.32138872 0.77458423]
    # truck: [1.77983356 0.25514606]

    resolution = f"{w}x{h}"
    for date_dir in ROOT_DIR.iterdir():
        if not date_dir.is_dir():
            continue
        for scene_dir in sorted(date_dir.glob("*sync")):
            for camera_number in camera_numbers:
                data_dir_tpl = str(scene_dir / ("{}_" + resolution + "_0" + camera_number))
                road_dir = Path(data_dir_tpl.format("road_segm"))
                # segms_labels_dir = Path(data_dir_tpl.format("segms_labels"))
                segms_labels_dir = Path(data_dir_tpl.format("modified_segms_labels_person_car"))
                disp_dir = Path(data_dir_tpl.format("disp"))
                img_dir = Path(data_dir_tpl.format("image"))
                print(img_dir)
                cam_heights = []

                basis_normal = None
                for road_idx, road_path in enumerate(tqdm(sorted(road_dir.glob("*npy")))):
                    # TODO: delete img
                    # img = np.array(Image.open(img_dir / f"{road_path.stem}.jpg"))
                    # img = cv2.LUT(img, img2gamma)

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
                    # nearest_pts_old = search_nearest_pts(eroded_segms, disp)
                    if erode_valid_idxs.shape[0] == 0:
                        continue

                    pix_heights = masks_to_pix_heights(segms)[erode_valid_idxs]
                    height_expects = height_priors[labels[erode_valid_idxs]][:, 0]
                    ideal_depths = height_expects / pix_heights * fy

                    # calc nearest point of each segm
                    cam_pts = depth2cam_pts(depth, cam_grid)
                    masked_cam_pts = cam_pts[None, ...] * eroded_segms[..., None]
                    masked_cam_pts[masked_cam_pts == 0] = 1000
                    nearest_pts = argmax_3d(-np.linalg.norm(masked_cam_pts, axis=3))
                    nearest_depths = depth[nearest_pts[:, 0], nearest_pts[:, 1]]

                    # TODO: meanで計算したらどうなるか？->これでいけるなら損失として設定するのでも希望が見えてくる
                    # mean_depths = np.zeros(eroded_segms.shape[0])
                    # for i, segm in enumerate(eroded_segms):
                    #     mean_depths[i] = depth[segm == 1].mean()

                    # plt.figure(figsize=(30, 5))
                    # plt.axis("off")
                    # plt.imshow(img)
                    # plt.plot(nearest_pts[:, 1], nearest_pts[:, 0], "ro", alpha=0.7, markersize=2)
                    # plt.plot(nearest_pts_old[:, 1], nearest_pts_old[:, 0], "wo", alpha=0.7, markersize=2)
                    # plt.tight_layout()
                    # plt.show()
                    # plt.close()

                    # FIXME: meanじゃなくてきちんとベイズで処理する
                    mean_scale = (ideal_depths / nearest_depths).mean()
                    # mean_scale = (ideal_depths / mean_depths).mean()

                    scaled_depth = mean_scale * depth
                    cam_pts = depth2cam_pts(scaled_depth, cam_grid)

                    # normal = cam_pts2normal(cam_pts, road_mask)
                    # if road_idx == 0 or basis_normal is None:
                    #     basis_normal = normal
                    # else:
                    #     R = calc_rot_mat(normal, basis_normal)

                    #     # d = np.pi / 60
                    #     # Rx = np.array([[1, 0, 0], [0, cos(d), -sin(d)], [0, sin(d), cos(d)]])
                    #     # Ry = np.array([[cos(d), 0, sin(d)], [0, 1, 0], [-sin(d), 0, cos(d)]])
                    #     # Rz = np.array([[cos(d), -sin(d), 0], [sin(d), cos(d), 0], [0, 0, 1]])
                    #     # R = Rz @ Ry

                    #     cam_pts_T = cam_pts.reshape(h * w, 3).T  # [3, h * w]
                    #     # rotated_cam_pts = (R @ cam_pts_T / cam_pts_T[2, :]).T.reshape(h, w, 3)
                    #     # rotated_cam_grid = (K @ R @ cam_pts_T / cam_pts_T[2, :])[:2, :].T.reshape(h, w, 2)
                    #     rotated_pix_grid = K @ R @ cam_pts_T  # [3, h * w]
                    #     # TODO: 画像平面よりも光学中心側に入ってる点は取り除くべき（たぶんあんまり問題にはならんけど）
                    #     rotated_pix_grid = (rotated_pix_grid / rotated_pix_grid[2, :])[:2, :]  # [2, h * w]

                    #     # HACK: rotationするとインデックスが元画像のサイズを飛び出すことに注意
                    #     # とりあえずクロップすることで対処
                    #     xs, ys = np.round(rotated_pix_grid).astype(np.int32)
                    #     inside_img_idxs = np.where((0 <= xs) & (xs < w) & (0 <= ys) & (ys < h))
                    #     # new_segms = np.zeros_like(segms, dtype=np.uint8)
                    #     # new_segms[:, ys[inside_img_idxs], xs[inside_img_idxs]] = segms.reshape(-1, h * w).T[inside_img_idxs].T
                    #     # new_segm = np.zeros((h, w), dtype=np.uint8)
                    #     # for i, segm in enumerate(new_segms):
                    #     #     new_segm[segm == 1] = i + 1
                    #     # plt.figure(figsize=(20, 5))
                    #     # plt.imshow(new_segm)
                    #     # plt.tight_layout()
                    #     # plt.axis("off")
                    #     # plt.show()
                    #     # plt.close()
                    #     plt.figure(figsize=(30, 5))
                    #     new_img = np.zeros_like(img, dtype=np.uint8)
                    #     new_img[ys[inside_img_idxs], xs[inside_img_idxs], :] = img.reshape(h * w, 3)[inside_img_idxs]
                    #     plt.subplot(1, 2, 1)
                    #     plt.imshow(new_img)
                    #     # plt.tight_layout()
                    #     # plt.axis("off")

                    #     plt.subplot(1, 2, 2)
                    #     # black_zone = (new_img.sum(axis=2) == 0).astype(np.uint8)
                    #     # plt.imshow(cv2.dilate(black_zone, erode_kernel))
                    #     alpha = 0.4
                    #     plt.imshow(cv2.addWeighted(new_img, alpha, img, 1 - alpha, 0))
                    #     # plt.imshow(new_img.sum(axis=2) - np.flip(img, axis=0).sum(axis=2))
                    #     plt.tight_layout()
                    #     plt.axis("off")
                    #     plt.show()
                    #     plt.close()

                    cam_height = cam_pts2cam_height(cam_pts, road_mask)
                    print(round(cam_height, 2), end=", ")
                    cam_heights.append(cam_height)
                print("")
                print(sum(cam_heights) / len(cam_heights))
                print("\n")


# print(cam_heights)
# FIXME: meanじゃなくてベイズ的に処理
# print(sum(cam_heights) / len(cam_heights))
# print(cam_height)

# normal = depth2normal(cam_pts)
# plot_normal_map(normal)

# img_path = img_dir / f"{road_path.stem}.jpg"
# img = Image.open(img_path)

# if segms.shape[0] > 1:
#     fig_w = len(th_list) // 2
#     fig, axes = plt.subplots(2, fig_w, figsize=(10 * len(th_list), 5), tight_layout=True)
#     for th_idx, th in enumerate(th_list):
#         tmp_segms = exclude_small_segm(segms, labels, th)[0]
#         for i, tmp_segm in enumerate(tmp_segms[1:]):
#             tmp_segms[0, tmp_segm == 1] = i + 1
#         axes[th_idx // fig_w, th_idx % fig_w].imshow(tmp_segms[0], cmap="Set1_r")
#         axes[th_idx // fig_w, th_idx % fig_w].axis("off")
#     axes[1, 2].imshow(img)
#     fig.tight_layout()
#     plt.show()
#     plt.close()
#     print(img_path)

# road_mask[road_mask == 1] = segm.max(axis=(0, 1))
# plt.figure(figsize=(20, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(road_mask + segm, cmap="tab20c")
# plt.tight_layout()
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(disp.reshape(h, w), cmap="magma")
# plt.tight_layout()
# plt.axis("off")
# plt.close()
