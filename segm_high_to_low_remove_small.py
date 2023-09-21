from concurrent import futures
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm

# from PIL import Image
# import matplotlib.pyplot as plt

ROOT_DIR = Path("/home/gkinoshita/workspace/monodepth2/kitti_data/")
AREA_TH = 400

high_res_str = "1024x320"
low_res = (192, 640)
low_res_str = "640x192"

resize = partial(resize, size=low_res, interpolation=v2.InterpolationMode.NEAREST_EXACT, antialias=False)

# camera_numbers = ("02", "03")
# for date_dir in tqdm(sorted(ROOT_DIR.iterdir())):
#     if not date_dir.is_dir():
#         continue
#     for scene_dir in tqdm(sorted(date_dir.glob("*sync"))):
#         for cam_num in camera_numbers:
#             highres_segm_dir = scene_dir / f"modified_segms_labels_person_car_road_{high_res_str}_{cam_num}"
#             highres_out_dir = scene_dir / f"modified_segms_labels_person_car_road_th400_{high_res_str}_{cam_num}"
#             highres_out_dir.mkdir(exist_ok=True, parents=True)
#             lowres_out_dir = scene_dir / f"modified_segms_labels_person_car_road_th400_{low_res_str}_{cam_num}"
#             lowres_out_dir.mkdir(exist_ok=True, parents=True)
#             # lowres_img_dir = scene_dir / f"image_{low_res_str}_{cam_num}"
#             for segm_path in highres_segm_dir.glob("*npz"):
#                 segms_labels = np.load(segm_path)
#                 segms_with_road = segms_labels["segms"] > 0
#                 road = segms_with_road[0:1] > 0
#                 segms = segms_with_road[1:]
#                 labels = segms_labels["labels"]

#                 valid_bools = segms.sum(axis=(1, 2)) > AREA_TH
#                 valid_segms = segms[valid_bools] > 0
#                 valid_labels = labels[valid_bools]
#                 valid_segms_with_road = np.concatenate((road, valid_segms), axis=0)
#                 out_path = highres_out_dir / segm_path.name
#                 np.savez(out_path, segms=valid_segms_with_road, labels=valid_labels)


#                 shrunk = resize(torch.from_numpy(valid_segms_with_road)).bool().numpy()
#                 out_path = lowres_out_dir / segm_path.name
#                 np.savez(out_path, segms=shrunk, labels=valid_labels)
def process(segm_path: Path, highres_out_dir: Path, lowres_out_dir: Path):
    segms_labels = np.load(segm_path)
    segms_with_road = segms_labels["segms"] > 0
    road = segms_with_road[0:1] > 0
    segms = segms_with_road[1:]
    labels = segms_labels["labels"]

    valid_bools = segms.sum(axis=(1, 2)) > AREA_TH
    valid_segms = segms[valid_bools] > 0
    valid_labels = labels[valid_bools]
    valid_segms_with_road = np.concatenate((road, valid_segms), axis=0)
    out_path = highres_out_dir / segm_path.name
    np.savez(out_path, segms=valid_segms_with_road, labels=valid_labels)

    shrunk = resize(torch.from_numpy(valid_segms_with_road)).bool().numpy()
    out_path = lowres_out_dir / segm_path.name
    np.savez(out_path, segms=shrunk, labels=valid_labels)


camera_numbers = ("02", "03")
for date_dir in tqdm(sorted(ROOT_DIR.iterdir())):
    if not date_dir.is_dir():
        continue
    for scene_dir in tqdm(sorted(date_dir.glob("*sync"), reverse=True)):
        print(scene_dir)
        for cam_num in camera_numbers:
            highres_segm_dir = scene_dir / f"modified_segms_labels_person_car_road_{high_res_str}_{cam_num}"
            highres_out_dir = scene_dir / f"modified_segms_labels_person_car_road_th400_{high_res_str}_{cam_num}"
            highres_out_dir.mkdir(exist_ok=True, parents=True)
            lowres_out_dir = scene_dir / f"modified_segms_labels_person_car_road_th400_{low_res_str}_{cam_num}"
            lowres_out_dir.mkdir(exist_ok=True, parents=True)
            with futures.ProcessPoolExecutor(max_workers=8) as executor:
                for segm_path in highres_segm_dir.glob("*npz"):
                    executor.submit(process, segm_path, highres_out_dir, lowres_out_dir)
