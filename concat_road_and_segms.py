from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_segms_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    segms, labels = npz["segms"], npz["labels"]
    return segms, labels


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segm_dir_prefix", type=str, default="modified_segms_labels_person_car")
    return parser.parse_args()


def concat_road_and_segms():
    add_humpback = False

    args = get_args()

    resolutions = ("1024x320",)
    camera_numbers = ("2", "3")
    kitti_root_dir = Path(f"/home/gkinoshita{'/humpback' if add_humpback else ''}/workspace/monodepth2/kitti_data")
    for date_dir in tqdm(sorted(kitti_root_dir.iterdir())):
        if not date_dir.is_dir():
            continue
        for scene_dir in tqdm(sorted(date_dir.glob(f"{date_dir.name}*"))):
            for resolution in resolutions:
                for camera_number in camera_numbers:
                    segms_camera_dir = scene_dir / f"{args.segm_dir_prefix}_{resolution}_0{camera_number}"
                    road_dir = scene_dir / f"road_segm_{resolution}_0{camera_number}"
                    if not segms_camera_dir.exists():
                        raise FileNotFoundError(f"{segms_camera_dir} does not exist.")
                    save_dir = scene_dir / f"{args.segm_dir_prefix}_road_{resolution}_0{camera_number}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    for segm_path in segms_camera_dir.glob("*.npz"):
                        segms, labels = load_segms_labels(segm_path)
                        road_mask = np.load(road_dir / f"{segm_path.stem}.npy")
                        segms = np.concatenate((road_mask[None, ...], segms), axis=0)
                        # labels = np.append(np.array(np.iinfo(np.uint8).max, dtype=np.uint8), labels)
                        # HACK: labelsには追加しない
                        np.savez(save_dir / segm_path.name, segms=segms, labels=labels)


if __name__ == "__main__":
    concat_road_and_segms()
