import argparse
from functools import cache
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


def remove_small_insts(segms: np.ndarray, labels: np.ndarray, area_th: int):
    areas = segms.sum(axis=(1, 2))
    enough_size_idxs = areas > area_th
    return segms[enough_size_idxs], labels[enough_size_idxs]


def median_abs_deviation(y: np.ndarray):
    mad = np.median(np.abs(y - np.median(y)))
    return mad if mad > 1 else 1


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


@cache
# def calc_vanishing_point(t0: np.ndarray, b0: np.ndarray, b: np.ndarray, r: np.ndarray, H: float, R: float) -> tuple[float, float] | None:
def calc_vanishing_point(
    t0: tuple[int, int], b0: tuple[int, int], b: tuple[int, int], r: tuple[int, int], H: float, R: float
) -> tuple[float, float] | None:
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted_ransac", action="store_true", help="whether weighing on RANSAC")
    parser.add_argument("--sampling_rate", type=float, help="This value is used as ``min_samples`` in ransac", default=0.1)
    parser.add_argument("--wo_corner", action="store_true", help="whether using segms without corner instances")
    parser.add_argument("--wo_small", action="store_true", help="whether using segms without small instances")
    parser.add_argument("--area_th", type=int, help="threshold of an instance area as small instances")
    parser.add_argument("--n_ransac_offset", type=int, default=4, help="the number of offset images for RANSAC")
    parser.add_argument("--ransac_min_inst", type=int, default=20, help="the min number of instances whether calculating RANSAC")
    parser.add_argument("--remove_th", type=float, default=0.6, help="threshold for regarding instances as outliers")
    parser.add_argument("--use_annot_height", action="store_true", help="whether using 3D BBox height labels for mean and var of car height")
    return parser.parse_args()


def main(args):
    add_humpback = False
    random_state = 42
    lower_bound_n_sample = 5
    upper_bound_n_sample = 20

    weighted_ransac = args.weighted_ransac
    wo_corner = args.wo_corner
    wo_small = args.wo_small
    area_th = args.area_th
    n_ransac_offset = args.n_ransac_offset
    ransac_min_inst = args.ransac_min_inst
    remove_th = args.remove_th

    resolutions = ("1024x320",)
    camera_numbers = ("2",)
    root_dir = Path(f"/home/gkinoshita/{'humpback/' if add_humpback else ''}workspace/monodepth2/kitti_data")
    if args.use_annot_height:
        str_annot_height = "_annot_height"
        print("use 3D BBox height labels")
        height_priors = np.array(
            [
                [1.747149944305419922e00, 6.863836944103240967e-02],
                [np.nan, np.nan],
                [1.5260834, 0.01868551],
            ]
        )
    else:
        str_annot_height = ""
        height_path = root_dir / "height_priors.txt"
        height_priors = np.loadtxt(height_path)

    kitti_horizon_dir = Path("/home/gkinoshita/workspace/kitti_horizon")
    root_save_dir = (
        kitti_horizon_dir
        / f"new_outliers_monodepth2_person_car{str_annot_height}{'_weighted' if weighted_ransac else ''}_sampling{args.sampling_rate}{'_wo_corner' if wo_corner else ''}{'_wo_small' if wo_small else ''}{area_th if wo_small and area_th > 0 else ''}_offset{n_ransac_offset}_th{str(remove_th).replace('.', '')}_min_inst{ransac_min_inst}"
    )
    root_save_dir.mkdir(parents=False, exist_ok=True)

    segm_dir_prefix = "modified_segms_labels_person_car"
    for date_dir in root_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for scene_dir in tqdm(sorted(date_dir.glob(f"{date_dir.name}*"))):
            for res in resolutions:
                img_w, img_h = map(int, res.split("x"))
                for cam_number in camera_numbers:
                    segm_dir = scene_dir / f"{segm_dir_prefix}{'_wo_corner' if wo_corner else ''}_{res}_0{cam_number}"
                    img_dir = scene_dir / f"image_{res}_0{cam_number}"
                    horizon_dir = scene_dir / f"horizon_{res}_0{cam_number}"
                    save_dir = root_save_dir / str(scene_dir).replace(str(root_dir), "")[1:] / f"{res}_0{cam_number}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    segm_paths = sorted(segm_dir.glob("*npz"))

                    cumsum_n_inst = [0] * len(segm_paths)
                    scene_n_inst = [0] * len(segm_paths)

                    scene_heights: list[np.ndarray] = [np.array([])] * len(segm_paths)
                    scene_tops: list[np.ndarray] = [np.array([])] * len(segm_paths)
                    scene_bottoms: list[np.ndarray] = [np.array([])] * len(segm_paths)
                    scene_segms: list[np.ndarray] = []

                    for f_idx in range(len(segm_paths)):
                        segm_path = segm_paths[f_idx]
                        segms, labels = load_segms_and_labels(segm_path)
                        if wo_small and area_th:
                            segms, labels = remove_small_insts(segms, labels, area_th)
                        scene_segms.append(segms)
                        n_inst = segms.shape[0]
                        scene_n_inst[f_idx] = n_inst
                        cumsum_n_inst[f_idx] = n_inst if f_idx == 0 else cumsum_n_inst[f_idx - 1] + n_inst

                        if n_inst > 0:
                            boxes = masks_to_boxes(segms)
                            rects = make_rects(boxes)

                            heights = height_priors[labels, 0]
                            mid_xs = (boxes[:, 0] + boxes[:, 2]) / 2
                            top_ys = img_h - boxes[:, 1]
                            bottom_ys = img_h - boxes[:, 3]
                            tops = np.stack((mid_xs, top_ys), axis=1)
                            bottoms = np.stack((mid_xs, bottom_ys), axis=1)
                            scene_heights[f_idx] = heights
                            scene_tops[f_idx] = tops
                            scene_bottoms[f_idx] = bottoms

                    for f_idx in range(len(segm_paths)):  # ここで物体が登場しなかったものがseqの１つとしてカウントされなかった場合，フレーム間が指定より飛びすぎてしまうのでよくない→
                        occur_cnts = np.zeros(scene_n_inst[f_idx], dtype=np.uint16)
                        outlier_cnts = np.copy(occur_cnts)
                        start_f_idx = max(0, f_idx - n_ransac_offset)
                        end_f_idx = f_idx + n_ransac_offset
                        seq_heights = np.concatenate(scene_heights[start_f_idx : end_f_idx + 1])

                        # scene_tops, scene_bottomsは [np.array([x, x])] なので単純にconcatできない
                        seq_tops = concat_removing_empties(scene_tops[start_f_idx : end_f_idx + 1])
                        seq_bottoms = concat_removing_empties(scene_bottoms[start_f_idx : end_f_idx + 1])

                        vp_x_y_lst = [[], []]
                        vp_idx2f_idxs: list[list[int, int]] = []
                        vp_idx2segm_idxs_in_frame: list[list[int, int]] = []

                        vp_weight_lst = []

                        # i, jがseq_segmsのうちどれに対応しているかを知る
                        # 累積和の[frame_idx + offset] - [frame_idx - offset]により，start_idx~end_idxまでの間にいくつのインスタンスがあるかわかる
                        # それによりi, jに対応するインスタンスがわかる(というよりもi, jをsegmsのindex(frame_idxとかを使う)に変換する)
                        #    i(, j)の大きさからどのフレームの何番目のインスタンスを見ているかがわかる
                        # やりたいのはvpとインスタンスを紐付けておいて，outlierになったvpに紐付いたインスタンスを理解すること

                        inst_idx2frame_idx = []
                        for f_idx_in_seq, n_inst in enumerate(scene_n_inst[start_f_idx : end_f_idx + 1]):
                            inst_idx2frame_idx += [f_idx_in_seq + start_f_idx] * n_inst

                        for i, j in combinations([k for k in range(seq_heights.shape[0])], 2):
                            # i のほうが大きい側
                            R = seq_heights[i]
                            H = seq_heights[j]
                            b = tuple(seq_bottoms[i].tolist())
                            b0 = tuple(seq_bottoms[j].tolist())
                            r = tuple(seq_tops[i].tolist())
                            t0 = tuple(seq_tops[j].tolist())
                            vp = calc_vanishing_point(t0, b0, b, r, H, R)
                            if vp is None:
                                continue

                            # start_idx ~ end_idxまでの範囲で表したときのframe_idx
                            i_f_idx_in_seq = inst_idx2frame_idx[i]
                            j_f_idx_in_seq = inst_idx2frame_idx[j]

                            interest_f_idx_in_seq = min(f_idx, n_ransac_offset)
                            vp_weight_lst.append(abs(i - interest_f_idx_in_seq) + abs(j - interest_f_idx_in_seq))

                            # i_f_idx_in_seqのうち何番目のインスタンスかを特定する
                            # cumsumを計算しといて，start_idx ~ i_frame_idx - 1の中にインスタンスがいくつあるのか知る.i - そのインスタンス数がframe_idxのうち何番目のインスタンスかに対応している
                            # n_inst_in_prev_k: seqの中で，obj_kの現れるフレームよりも前のフレームに存在する物体の個数の和
                            if i_f_idx_in_seq == 0:
                                n_inst_in_prev_i = 0
                            elif start_f_idx == 0:
                                n_inst_in_prev_i = cumsum_n_inst[i_f_idx_in_seq - 1]
                            else:
                                n_inst_in_prev_i = cumsum_n_inst[i_f_idx_in_seq - 1] - cumsum_n_inst[start_f_idx - 1]

                            if j_f_idx_in_seq == 0:
                                n_inst_in_prev_j = 0
                            elif start_f_idx == 0:
                                n_inst_in_prev_j = cumsum_n_inst[j_f_idx_in_seq - 1]
                            else:
                                n_inst_in_prev_j = cumsum_n_inst[j_f_idx_in_seq - 1] - cumsum_n_inst[start_f_idx - 1]
                            vp_x_y_lst[0].append(vp[0])
                            vp_x_y_lst[1].append(img_h - vp[1])  # imshowのときはy軸が左上原点になる. RANSACとしてはどうでもいい

                            vp_idx2f_idxs.append([i_f_idx_in_seq, j_f_idx_in_seq])
                            vp_idx2segm_idxs_in_frame.append([i - n_inst_in_prev_i, j - n_inst_in_prev_j])

                            if i_f_idx_in_seq == f_idx:
                                occur_cnts[i - n_inst_in_prev_i] += 1
                            if j_f_idx_in_seq == f_idx:
                                occur_cnts[j - n_inst_in_prev_j] += 1

                        coef = None
                        intercept = None

                        # 可能性としては，min_samplesで 10 * min_samples

                        """
                        min_samples=0.1 | 0.25
                        X.shape[0] >= 20 (ransac_min_inst)
                        X.shape[0] * min_samples >= 20 * 0.1 == 2 | 20 * 0.25 == 5
                        -> min_samples = 5 (lower_bound_n_sample)

                        Xまたはyの値が似通ってる時にエラーが出る？
                        """
                        if len(vp_x_y_lst[0]) >= ransac_min_inst:
                            min_samples = args.sampling_rate
                            X = np.array(vp_x_y_lst[0]).reshape(-1, 1)
                            y = np.array(vp_x_y_lst[1]).reshape(-1, 1)
                            if X.shape[0] * min_samples < lower_bound_n_sample:
                                min_samples = lower_bound_n_sample
                            elif X.shape[0] * min_samples > upper_bound_n_sample:
                                min_samples = upper_bound_n_sample
                            # print(f"{min_samples=}, {X.shape[0] * min_samples=}")
                            # TODO: vpのyの値にばらつきが大きい時にresidual_thresholdが大きくなり，n_trialsが少なくなることがある．たぶん影響は小さいので後回し．
                            ransac = linear_model.RANSACRegressor(
                                random_state=random_state, min_samples=min_samples, residual_threshold=median_abs_deviation(y)
                            )
                            # HACK: ransac.residual_thresholdが小さすぎてinlierが見つからない可能性がある．今はthが1以上になるようにしているがそれでもinlierが見つからなかったら全instanceがoutlier or 下限値を上げる
                            try:
                                if weighted_ransac:
                                    ransac.fit(X, y, sample_weight=vp_weight_lst)
                                else:
                                    ransac.fit(X, y)
                            except Exception as e:
                                print(f"\n{e}\n")
                                print(f"{ransac.residual_threshold=}")

                            # print(f"{ransac.n_trials_=}, {median_abs_deviation(y)=:.1f}")
                            outlier_mask = np.logical_not(ransac.inlier_mask_)

                            vp_idx2f_idxs = np.array(vp_idx2f_idxs)
                            vp_idx2segm_idxs_in_frame = np.array(vp_idx2segm_idxs_in_frame)

                            outlier_vp_idx2f_idxs = vp_idx2f_idxs[outlier_mask]
                            outlier_vp_idx2segm_idxs = vp_idx2segm_idxs_in_frame[outlier_mask]

                            for outlier_f_idxs, outlier_segm_idxs in zip(outlier_vp_idx2f_idxs, outlier_vp_idx2segm_idxs):
                                if outlier_f_idxs[0] == f_idx:
                                    outlier_cnts[outlier_segm_idxs[0]] += 1
                                if outlier_f_idxs[1] == f_idx:
                                    outlier_cnts[outlier_segm_idxs[1]] += 1

                            coef = ransac.estimator_.coef_.item()
                            intercept = ransac.estimator_.intercept_.item()

                        plot_idxs = None
                        lw = 1
                        figsize = (15, 6)
                        img_path = img_dir / f"{segm_paths[f_idx].stem}.jpg"
                        img = np.array(Image.open(img_path))
                        horizon_path = horizon_dir / f"{img_path.stem}.npy"
                        horizon = np.load(horizon_path)  # x1, x2, y1, y2
                        n_inst = scene_segms[f_idx].shape[0]
                        if n_inst > 0:
                            delta = 0.1
                            segmented_img = colorize_segment(img, scene_segms[f_idx])
                            if coef is None:
                                boxes = masks_to_boxes(scene_segms[f_idx])
                            else:
                                if coef == 0:
                                    x_arr = np.array((0, img_w - 1), dtype=np.float32)
                                else:
                                    x_arr = np.array(
                                        sorted((0, (delta - intercept) / coef, (img_h - delta - intercept) / coef, img_w - 1)), dtype=np.float32
                                    )
                                y_arr = coef * x_arr + intercept
                                plot_idxs = np.where((0 <= x_arr) & (x_arr <= img_w - 1) & (0 <= y_arr) & (y_arr <= img_h - 1))
                                occur_cnts[occur_cnts == 0] = 1
                                outlier_cnts[occur_cnts == 0] = LARGE_VALUE
                                outlier_inst_bools = outlier_cnts / occur_cnts > remove_th
                                boxes = masks_to_boxes(scene_segms[f_idx][outlier_inst_bools])
                            rects = make_rects(boxes)

                            plt.figure(figsize=figsize)
                            ax = plt.axes()
                            ax.set_title(f"{img_dir.name} frame {f_idx}", fontsize=10)
                            ax.imshow(segmented_img)
                            for i, rect in enumerate(np.array(rects)):
                                ax.add_patch(rect)
                            ax.plot(horizon[:2], horizon[2:], "r-", lw=lw, label="VL")
                            ax.plot([0, img_w - 1], [img_h / 2, img_h / 2], "w-", lw=lw, label="middle")

                            if plot_idxs is not None:
                                ax.plot(x_arr[plot_idxs], y_arr[plot_idxs], lw=lw, label="RANSAC", c="cyan")

                            ax.legend(loc="upper right", borderaxespad=0, fontsize=7)
                            ax.axis("off")
                            plt.savefig(save_dir / f"{str(f_idx).zfill(4)}.jpg", bbox_inches="tight", pad_inches=0)
                            # plt.tight_layout()
                            # plt.show()
                            plt.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
