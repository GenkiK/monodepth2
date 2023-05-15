import argparse
from copy import deepcopy
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn import linear_model
from tqdm import tqdm


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


def load_segms_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    segms = npz["segms"].astype(np.uint8)
    labels = npz["labels"].astype(np.uint8)
    return segms, labels


@cache
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


def concat_removing_empties(arr_lst: list[np.ndarray]) -> np.ndarray:
    new_arr_lst = [arr for arr in arr_lst if arr.shape[0] > 0]
    if len(new_arr_lst):
        return np.concatenate(new_arr_lst)
    else:
        return np.array([], dtype=np.float32)


LARGE_VALUE = np.iinfo(np.uint16).max

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_annot_height", help="whether using 3D BBox height labels for mean and var of car height", action="store_true")
    args = parser.parse_args()

    use_3DBBox_annot_height = args.use_annot_height
    n_ransac_offset = 4
    remove_th = 0.6
    th_str = str(remove_th).replace(".", "")
    min_inst = 10

    resolutions = ("1024x320",)
    camera_numbers = ("2", "3")
    root_dir = Path("/home/gkinoshita/workspace/monodepth2/kitti_data")
    if use_3DBBox_annot_height:
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

    segm_dir_prefix = "modified_segms_labels_person_car_wo_corner"

    for date_dir in root_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for scene_dir in tqdm(sorted(date_dir.glob(f"{date_dir.name}*"))):
            for res in resolutions:
                img_w, img_h = map(int, res.split("x"))
                for cam_number in camera_numbers:
                    segm_dir = scene_dir / f"{segm_dir_prefix}_{res}_0{cam_number}"
                    save_dir = (
                        scene_dir / f"{segm_dir_prefix}{str_annot_height}_offset{n_ransac_offset}_th{th_str}_min_inst{min_inst}_{res}_0{cam_number}"
                    )
                    save_dir.mkdir(parents=False, exist_ok=True)

                    segm_paths = sorted(segm_dir.glob("*npz"))

                    cumsum_n_inst = [0] * len(segm_paths)
                    scene_n_inst = [0] * len(segm_paths)

                    scene_heights: list[np.ndarray] = [np.array([])] * len(segm_paths)
                    scene_tops: list[np.ndarray] = [np.array([])] * len(segm_paths)
                    scene_bottoms: list[np.ndarray] = [np.array([])] * len(segm_paths)

                    for f_idx in range(len(segm_paths)):
                        segm_path = segm_paths[f_idx]
                        segms, labels = load_segms_and_labels(segm_path)

                        n_inst = segms.shape[0]
                        scene_n_inst[f_idx] = n_inst
                        cumsum_n_inst[f_idx] = n_inst if f_idx == 0 else cumsum_n_inst[f_idx - 1] + n_inst

                        if n_inst > 0:
                            boxes = masks_to_boxes(segms)
                            heights = height_priors[labels, 0]
                            mid_xs = (boxes[:, 0] + boxes[:, 2]) / 2
                            top_ys = img_h - boxes[:, 1]
                            bottom_ys = img_h - boxes[:, 3]
                            tops = np.stack((mid_xs, top_ys), axis=1)
                            bottoms = np.stack((mid_xs, bottom_ys), axis=1)
                            scene_heights[f_idx] = heights
                            scene_tops[f_idx] = tops
                            scene_bottoms[f_idx] = bottoms
                        # else: scene_xxx[f_idx] = np.array([])

                    # 各物体の登場回数(計算に使われた回数)
                    scene_segm_occur_cnts = [np.zeros(n_inst, dtype=np.uint16) for n_inst in scene_n_inst]
                    # outlierと認定された登場回数
                    scene_segm_outlier_cnts = deepcopy(scene_segm_occur_cnts)

                    ransac = linear_model.RANSACRegressor()
                    for f_idx in range(len(segm_paths)):  # ここで物体が登場しなかったものがseqの１つとしてカウントされなかった場合，フレーム間が指定より飛びすぎてしまうのでよくない
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

                        inst_idx2frame_idx = []
                        for f_idx_in_seq, n_inst in enumerate(scene_n_inst[start_f_idx : end_f_idx + 1]):
                            inst_idx2frame_idx += [f_idx_in_seq + start_f_idx] * n_inst

                        if f_idx == 0:
                            start_obj_idx_in_seq = 0
                        elif start_f_idx == 0:
                            start_obj_idx_in_seq = cumsum_n_inst[f_idx - 1]
                        else:
                            start_obj_idx_in_seq = cumsum_n_inst[f_idx - 1] - cumsum_n_inst[start_f_idx - 1]
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
                            vp_x_y_lst[1].append(img_h - vp[1])  # imshowのときはy軸が左上原点になる. RANSACとしてはどうでもいい

                            vp_idx2f_idxs.append([i_f_idx, j_f_idx])
                            vp_idx2segm_idxs_in_frame.append([i - n_inst_in_prev_i, j - n_inst_in_prev_j])

                            scene_segm_occur_cnts[i_f_idx][i - n_inst_in_prev_i] += 1
                            scene_segm_occur_cnts[j_f_idx][j - n_inst_in_prev_j] += 1

                        if len(vp_x_y_lst[0]) >= min_inst:
                            ransac.fit(np.array(vp_x_y_lst[0]).reshape(-1, 1), np.array(vp_x_y_lst[1]).reshape(-1, 1))

                            vp_idx2f_idxs = np.array(vp_idx2f_idxs)
                            vp_idx2segm_idxs_in_frame = np.array(vp_idx2segm_idxs_in_frame)

                            outlier_mask = np.logical_not(ransac.inlier_mask_)

                            seq_outlier_f_idxs = vp_idx2f_idxs[outlier_mask]
                            seq_outlier_segm_idxs_in_frame = vp_idx2segm_idxs_in_frame[outlier_mask]

                            for outlier_f_idxs, outlier_segm_idxs_in_frame in zip(seq_outlier_f_idxs, seq_outlier_segm_idxs_in_frame):
                                scene_segm_outlier_cnts[outlier_f_idxs[0]][outlier_segm_idxs_in_frame[0]] += 1
                                scene_segm_outlier_cnts[outlier_f_idxs[1]][outlier_segm_idxs_in_frame[1]] += 1

                    for f_idx in range(len(segm_paths)):
                        # HACK: 各フレームの物体のうち，VP計算に使われなかった物体は，除外する
                        scene_segm_occur_cnts[f_idx][scene_segm_occur_cnts[f_idx] == 0] = 1
                        scene_segm_outlier_cnts[f_idx][scene_segm_occur_cnts[f_idx] == 0] = LARGE_VALUE
                        inlier_segm_bools = scene_segm_outlier_cnts[f_idx] / scene_segm_occur_cnts[f_idx] <= remove_th

                        segms, labels = load_segms_and_labels(segm_paths[f_idx])
                        save_path = save_dir / segm_paths[f_idx].name
                        np.savez(save_path, segms=segms[inlier_segm_bools], labels=labels[inlier_segm_bools])
