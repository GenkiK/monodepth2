from copy import deepcopy
from itertools import combinations
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn import linear_model
from tqdm import tqdm


def load_segms_and_labels(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    npz = np.load(path)
    segms = npz["segms"].astype(np.uint8)
    labels = npz["labels"].astype(np.uint8)
    return segms, labels


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    n = masks.shape[0]
    bboxes = np.zeros((n, 4), dtype=np.uint16)
    for idx, mask in enumerate(masks):
        y, x = np.where(mask != 0)
        bboxes[idx, 0] = np.min(x)
        bboxes[idx, 1] = np.min(y)
        bboxes[idx, 2] = np.max(x)
        bboxes[idx, 3] = np.max(y)
    return bboxes


cmap = plt.get_cmap("Set3")
LARGE_VALUE = np.iinfo(np.uint16).max


def make_rects(boxes: torch.Tensor) -> tuple[torch.Tensor, list[patches.Rectangle]]:
    rects = []
    for idx, box in enumerate(boxes):
        # color = [int(255 * c) for c in cmap(idx)[:3]]
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=cmap(idx)[:3],
            facecolor="none",
            lw=2.5,
        )
        rects.append(rect)
    return rects


def colorize_segment(img, segm):
    segmented_imgs = img.copy()
    alpha = 0.5
    for inst_idx in range(segm.shape[0]):
        tmp_masks = segm[inst_idx, :, :] == 1
        segmented_imgs[tmp_masks] = segmented_imgs[tmp_masks] * alpha + np.array(
            [int(255 * color) for color in cmap(inst_idx)[:3]], dtype=np.uint8
        ) * (1 - alpha)
    return segmented_imgs


def calc_vanishing_point(t0: np.ndarray, b0: np.ndarray, b: np.ndarray, r: np.ndarray, H: float, R: float) -> tuple[float, float] | None:
    a0 = b0[0]
    c0 = t0[1]
    a1 = b[0]
    b1 = b[1]
    d1 = r[1]
    c1 = abs(H / R * (d1 - b1) ** 2 * (a1**2 + 1) / (a1**2 + 1 + (d1 - b1) ** 2 - H / R * (d1 - b1) ** 2)) ** 0.5 + b1
    if c1 - c0 - (b[1] - b0[1]) == 0 or b[0] - b0[0] == 0:
        return None
    m = (b[1] - b0[1]) / (b[0] - b0[0])
    n = -m * a1 + b1
    x = (a0 * (c1 - c0) + (n - c0) * (a1 - a0)) / (c1 - c0 - (b[1] - b0[1]))
    y = m * x + n
    return x, y


def interp_nan(arr):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    nans, f = np.isnan(arr), lambda x: x.nonzero()[0]
    arr[nans] = np.interp(f(nans), f(~nans), arr[~nans])
    return arr


def concat_removing_empties(arr_lst: list[np.ndarray]) -> np.ndarray:
    new_arr_lst = [arr for arr in arr_lst if arr.shape[0] > 0]
    if len(new_arr_lst):
        return np.concatenate(new_arr_lst)
    else:
        return np.array([], dtype=np.float32)


if __name__ == "__main__":
    is_humpback = False
    random_state = 42
    n_show_img = None
    n_ransac_offset = 4
    remove_th = 0.6
    ransac_min_inst = 10

    root_dir = Path(f"/home/gkinoshita/{'humpback/' if is_humpback else ''}workspace/monodepth2/kitti_data")
    height_path = root_dir / "height_priors.txt"
    height_priors = np.loadtxt(height_path)
    # FIXME: uncomment below codes
    resolutions = ("1024x320",)
    camera_numbers = ("2",)

    root_save_dir = Path(f"./outliers_monodepth2_person_car_offset{n_ransac_offset}_th{str(remove_th).replace('.', '')}_min_inst{ransac_min_inst}")
    root_save_dir.mkdir(parents=False, exist_ok=True)

    for date_dir in root_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for scene_dir in tqdm(sorted(date_dir.glob(f"{date_dir.name}*"))):
            for resolution in resolutions:
                img_w, img_h = map(int, resolution.split("x"))
                for camera_number in camera_numbers:
                    # segm_dir = scene_dir / f"segms_labels_{resolution}_0{camera_number}"
                    segm_dir = scene_dir / f"modified_segms_labels_person_car_{resolution}_0{camera_number}"
                    img_dir = scene_dir / f"image_{resolution}_0{camera_number}"
                    horizon_dir = scene_dir / f"horizon_{resolution}_0{camera_number}"
                    save_dir = root_save_dir / str(scene_dir).replace(str(root_dir), "")[1:] / f"{resolution}_0{camera_number}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    img_paths = sorted(img_dir.glob("*jpg"))

                    eff_n_show_img = len(img_paths) if n_show_img is None else min(n_show_img, len(img_paths))

                    cumsum_n_inst = []
                    scene_n_inst = []

                    scene_imgs = []
                    scene_horizons = []
                    scene_rects = []
                    scene_heights: list[np.ndarray] = []
                    scene_tops: list[np.ndarray] = []
                    scene_bottoms: list[np.ndarray] = []

                    for f_idx in range(eff_n_show_img):
                        # 表示のためだけの画像/horizon読み込み
                        img_path = img_paths[f_idx]
                        img = np.array(Image.open(img_path))
                        horizon_path = horizon_dir / f"{img_path.stem}.npy"

                        if horizon_path.exists():
                            horizon = np.load(horizon_path)  # x1, x2, y1, y2
                            scene_horizons.append(horizon)
                        else:
                            scene_horizons.append(None)

                        segm_path = segm_dir / f"{img_path.stem}.npz"
                        segms, labels = load_segms_and_labels(segm_path)

                        n_inst = segms.shape[0]
                        scene_n_inst.append(n_inst)
                        if f_idx == 0:
                            cumsum_n_inst.append(n_inst)
                        else:
                            cumsum_n_inst.append(cumsum_n_inst[f_idx - 1] + n_inst)

                        if n_inst > 0:
                            boxes = masks_to_boxes(segms)
                            segmented_img = colorize_segment(img, segms)
                            scene_imgs.append(segmented_img)
                            rects = make_rects(boxes)
                            scene_rects.append(rects)

                            seq_heights = height_priors[labels, 0]
                            mid_xs = (boxes[:, 0] + boxes[:, 2]) / 2
                            top_ys = img_h - boxes[:, 1]
                            bottom_ys = img_h - boxes[:, 3]
                            seq_tops = np.stack((mid_xs, top_ys), axis=1)
                            seq_bottoms = np.stack((mid_xs, bottom_ys), axis=1)
                            scene_heights.append(seq_heights)
                            scene_tops.append(seq_tops)
                            scene_bottoms.append(seq_bottoms)
                        else:
                            # print(f"{img_path} has no instance.")
                            scene_imgs.append(img)
                            scene_rects.append(np.array([]))
                            scene_heights.append(np.array([]))
                            scene_tops.append(np.array([]))
                            scene_bottoms.append(np.array([]))

                    # 各物体の登場回数(計算に使われた回数)
                    segm_occur_cnts = [np.zeros(n_inst, dtype=np.uint16) for n_inst in scene_n_inst]
                    # outlierと認定された登場回数
                    segm_outlier_cnts = deepcopy(segm_occur_cnts)

                    ransac = linear_model.RANSACRegressor(random_state=random_state)
                    coefs: list[float] = []
                    intercepts: list[float] = []
                    for f_idx in range(len(scene_imgs)):  # ここで物体が登場しなかったものがseqの１つとしてカウントされなかった場合，フレーム間が指定より飛びすぎてしまうのでよくない→
                        start_f_idx = max(0, f_idx - n_ransac_offset)
                        end_f_idx = f_idx + n_ransac_offset
                        seq_heights = np.concatenate(scene_heights[start_f_idx : end_f_idx + 1])

                        # scene_tops, scene_bottomsは [np.array([x, x])] なので単純にconcatできない
                        seq_tops = concat_removing_empties(scene_tops[start_f_idx : end_f_idx + 1])
                        seq_bottoms = concat_removing_empties(scene_bottoms[start_f_idx : end_f_idx + 1])

                        vp_x_y_lst = [[], []]
                        vp_idx2f_idxs: list[list[int, int]] = []
                        vp_idx2segm_idxs_in_frame: list[list[int, int]] = []

                        # i, jがseq_segmsのうちどれに対応しているかを知る
                        # 累積和の[frame_idx + offset] - [frame_idx - offset]により，start_idx~end_idxまでの間にいくつのインスタンスがあるかわかる
                        # それによりi, jに対応するインスタンスがわかる(というよりもi, jをsegmsのindex(frame_idxとかを使う)に変換する)
                        #    i(, j)の大きさからどのフレームの何番目のインスタンスを見ているかがわかる
                        # やりたいのはvpとインスタンスを紐付けておいて，outlierになったvpに紐付いたインスタンスを理解すること

                        # 一時的なprefix_sumを作る
                        # inst_prefix_sum = list(accumulate(seq_n_inst[start_idx : end_idx + 1]))
                        # これを[0, 0, 1, 1, 1, 2, 3, 4, 4]みたいにするとbisectする必要がなくなる
                        # indexがオフセットフレーム全体のうちのインスタンスidx(order), valueはそのインスタンスが存在するframeのidx(オフセットフレーム中のframe_idx)
                        inst_idx2frame_idx = []
                        for f_idx_in_seq, n_inst in enumerate(scene_n_inst[start_f_idx : end_f_idx + 1]):
                            inst_idx2frame_idx += [f_idx_in_seq + start_f_idx] * n_inst

                        if f_idx == 0:
                            start_obj_idx_in_seq = 0
                        elif start_f_idx == 0:
                            start_obj_idx_in_seq = cumsum_n_inst[f_idx - 1]
                        else:
                            start_obj_idx_in_seq = cumsum_n_inst[f_idx - 1] - cumsum_n_inst[start_f_idx - 1]
                        # for i in range(start_obj_idx_in_seq, start_obj_idx_in_seq + scene_n_inst[f_idx]):
                        # for j in range(seq_heights.shape[0]):
                        for i, j in combinations([k for k in range(seq_heights.shape[0])], 2):
                            # i のほうが大きい側
                            R = seq_heights[i]
                            H = seq_heights[j]
                            b = seq_bottoms[i]
                            b0 = seq_bottoms[j]
                            r = seq_tops[i]
                            t0 = seq_tops[j]
                            vp = calc_vanishing_point(t0, b0, b, r, H, R)
                            if vp is None:
                                continue

                            # start_idx ~ end_idxまでの範囲で表したときのframe_idx
                            i_f_idx = inst_idx2frame_idx[i]
                            j_f_idx = inst_idx2frame_idx[j]

                            # i_f_idxのうち何番目のインスタンスかを特定する
                            # cumsumを計算しといて，start_idx ~ i_frame_idx - 1の中にインスタンスがいくつあるのか知る.i - そのインスタンス数がframe_idxのうち何番目のインスタンスかに対応している
                            # n_inst_in_prev_k: seqの中で，obj_kの現れるフレームよりも前のフレームに存在する物体の個数の和
                            if i_f_idx == 0:
                                n_inst_in_prev_i = 0
                            elif start_f_idx == 0:
                                n_inst_in_prev_i = cumsum_n_inst[i_f_idx - 1]
                            else:
                                n_inst_in_prev_i = cumsum_n_inst[i_f_idx - 1] - cumsum_n_inst[start_f_idx - 1]

                            if j_f_idx == 0:
                                n_inst_in_prev_j = 0
                            elif start_f_idx == 0:
                                n_inst_in_prev_j = cumsum_n_inst[j_f_idx - 1]
                            else:
                                n_inst_in_prev_j = cumsum_n_inst[j_f_idx - 1] - cumsum_n_inst[start_f_idx - 1]
                            vp_x_y_lst[0].append(vp[0])
                            vp_x_y_lst[1].append(img_h - vp[1])  # imshowのときはy軸が左上原点になる

                            vp_idx2f_idxs.append([i_f_idx, j_f_idx])
                            vp_idx2segm_idxs_in_frame.append([i - n_inst_in_prev_i, j - n_inst_in_prev_j])

                            segm_occur_cnts[i_f_idx][i - n_inst_in_prev_i] += 1
                            segm_occur_cnts[j_f_idx][j - n_inst_in_prev_j] += 1

                        if len(vp_x_y_lst[0]) >= ransac_min_inst:
                            ransac.fit(np.array(vp_x_y_lst[0]).reshape(-1, 1), np.array(vp_x_y_lst[1]).reshape(-1, 1))

                            vp_idx2f_idxs = np.array(vp_idx2f_idxs)
                            vp_idx2segm_idxs_in_frame = np.array(vp_idx2segm_idxs_in_frame)

                            outlier_mask = np.logical_not(ransac.inlier_mask_)

                            seq_outlier_f_idxs = vp_idx2f_idxs[outlier_mask]
                            seq_outlier_segm_idxs_in_frame = vp_idx2segm_idxs_in_frame[outlier_mask]

                            for outlier_f_idxs, outlier_segm_idxs_in_frame in zip(seq_outlier_f_idxs, seq_outlier_segm_idxs_in_frame):
                                segm_outlier_cnts[outlier_f_idxs[0]][outlier_segm_idxs_in_frame[0]] += 1
                                segm_outlier_cnts[outlier_f_idxs[1]][outlier_segm_idxs_in_frame[1]] += 1

                            coefs.append(ransac.estimator_.coef_.item())
                            intercepts.append(ransac.estimator_.intercept_.item())
                        else:
                            coefs.append(np.nan)
                            intercepts.append(np.nan)

                    lw = 1
                    figsize = (15, 6)
                    coefs = np.array(coefs)
                    intercepts = np.array(intercepts)

                    if not np.isnan(coefs).all():
                        # horizonが決定できなかった場合に隣接フレームで滑らかになるように補完
                        coefs = interp_nan(coefs)
                        intercepts = interp_nan(intercepts)

                        for f_idx in range(len(scene_imgs)):
                            if len(segm_occur_cnts[f_idx]) == 0:
                                continue
                            segm_occur_cnts[f_idx][segm_occur_cnts[f_idx] == 0] = LARGE_VALUE
                            outlier_segm_idxs = segm_outlier_cnts[f_idx] / segm_occur_cnts[f_idx] > remove_th

                            plt.figure(figsize=figsize)
                            ax = plt.axes()
                            ax.set_title(f"{img_dir.name}  frame {f_idx}", fontsize=10)
                            ax.imshow(scene_imgs[f_idx])
                            for i, rect in enumerate(np.array(scene_rects[f_idx])[outlier_segm_idxs]):
                                ax.add_patch(rect)
                            horizon = scene_horizons[f_idx]
                            if horizon is not None:
                                ax.plot(horizon[:2], horizon[2:], "r-", lw=lw, label="VL")
                            ax.plot([0, img_w - 1], [img_h / 2, img_h / 2], "w-", lw=lw, label="middle")

                            x_arr = np.arange(0, img_w)

                            coef = coefs[f_idx]
                            intercept = intercepts[f_idx]
                            delta = 0.1
                            if coef == 0:
                                x_arr = np.array((0, img_w - 1), dtype=np.float32)
                            else:
                                x_arr = np.array(
                                    sorted((0, (delta - intercept) / coef, (img_h - delta - intercept) / coef, img_w - 1)), dtype=np.float32
                                )
                            y_arr = coef * x_arr + intercept
                            plot_idxs = np.where((0 <= x_arr) & (x_arr <= img_w - 1) & (0 <= y_arr) & (y_arr <= img_h - 1))
                            ax.plot(x_arr[plot_idxs], y_arr[plot_idxs], lw=lw, label="RANSAC", c="cyan")

                            ax.legend(loc="upper right", borderaxespad=0, fontsize=7)
                            ax.axis("off")
                            plt.savefig(save_dir / f"{str(f_idx).zfill(4)}.jpg", bbox_inches="tight", pad_inches=0)
                            # plt.tight_layout()
                            # plt.show()
                            plt.close()
                    else:
                        for f_idx in range(len(scene_imgs)):
                            plt.figure(figsize=figsize)
                            ax = plt.axes()
                            ax.set_title(f"{img_dir.name}  frame {f_idx}", fontsize=10)
                            ax.imshow(scene_imgs[f_idx])
                            for i, rect in enumerate(scene_rects[f_idx]):
                                ax.add_patch(rect)
                            horizon = scene_horizons[f_idx]
                            if horizon is not None:
                                ax.plot(horizon[:2], horizon[2:], "r-", lw=lw, label="VL")
                            ax.plot([0, img_w], [img_h / 2, img_h / 2], "w-", lw=lw, label="middle")
                            ax.legend(loc="upper right", borderaxespad=0, fontsize=7)
                            ax.axis("off")
                            plt.savefig(save_dir / f"{str(f_idx).zfill(4)}.jpg", bbox_inches="tight", pad_inches=0)
                            # plt.tight_layout()
                            # plt.show()
                            plt.close()
