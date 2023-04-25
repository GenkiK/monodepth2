from pathlib import Path

import numpy as np
import pykitti
from skimage import transform
from tqdm import tqdm


class KITTIHorizonRawMonodepth:
    def __init__(
        self,
        dataset_path: Path,
        resize_width: int,
        resize_height: int,
        # save_dir: Path = Path("/home/gkinoshita/dugong/workspace/Insta-DM/kitti_256/horizon"),
    ):
        self.basedir = dataset_path
        self.resize_width = resize_width
        self.resize_height = resize_height
        # self.save_dir = save_dir

    def get_date_ids(self):
        dates = []
        for entry in self.basedir.iterdir():
            if entry.is_dir():
                dates.append(entry.name)
        return dates

    def get_drive_ids(self, date: str):
        drives = []
        date_dir = self.basedir / date
        for entry in date_dir.iterdir():
            if entry.is_dir():
                drive = entry.name.split("_")[-2]
                drives.append(drive)
        return drives

    def get_drive(self, date_id: int, drive_id: int):
        dataset = pykitti.raw(self.basedir, date_id, drive_id)
        return dataset

    def process_single_image(self, drive: pykitti.raw, image: np.ndarray, idx: int, camera: int):
        if camera == 2:
            R_cam_imu = np.matrix(drive.calib.T_cam2_imu[0:3, 0:3])
            K = np.matrix(drive.calib.P_rect_20[0:3, 0:3])
        elif camera == 3:
            R_cam_imu = np.matrix(drive.calib.T_cam3_imu[0:3, 0:3])
            K = np.matrix(drive.calib.P_rect_30[0:3, 0:3])
        G = np.matrix([[0.0], [0.0], [1.0]])

        orig_image_width = image.width
        orig_image_height = image.height

        image = np.array(image)
        image = transform.resize(image, (self.resize_height, self.resize_width)).transpose(2, 0, 1)

        R_imu = np.matrix(drive.oxts[idx].T_w_imu[0:3, 0:3])
        G_imu = R_imu.T * G
        G_cam = R_cam_imu * G_imu

        h = np.array(K.I.T * G_cam).squeeze()

        hp1 = np.cross(h, np.array([1, 0, 0]))
        hp2 = np.cross(h, np.array([1, 0, -orig_image_width]))
        hp1 /= hp1[2]
        hp2 /= hp2[2]

        hp1[0] *= self.resize_height / orig_image_height
        hp2[0] *= self.resize_width / orig_image_width
        hp1[1] *= self.resize_height / orig_image_height
        hp2[1] *= self.resize_width / orig_image_width

        h = np.cross(hp1, hp2)

        angle = np.arctan2(h[0], h[1])
        if angle > np.pi / 2:
            angle -= np.pi
        elif angle < -np.pi / 2:
            angle += np.pi

        h = h / np.linalg.norm(h[0:2])
        # save_path = self.save_dir / Path(drive.drive + f"_0{camera}/{str(idx).zfill(10)}.txt")
        # save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "image": image,
            "horizon_hom": h,
            "horizon_p1": hp1,
            "horizon_p2": hp2,
            "angle": angle,
            "G": G_cam,
            "K": K,
            # "save_path": save_path,
        }
        return data


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Shows images from the raw KITTI dataset with horizons")
    parser.add_argument(
        "--path",
        default="/home/gkinoshita/humpback/dataset/packnet-kitti-raw/KITTI_raw",
        type=Path,
        help="path to KITTI rawdata",
    )
    parser.add_argument("--whole-data", action="store_true", help="")
    parser.add_argument("--date", default=None, type=str, help="")
    parser.add_argument("--drive", default=None, type=str, help="")
    parser.add_argument("--camera", default=2, type=int, choices=[2, 3])
    parser.add_argument("--resize_width", default=1024, type=int)
    parser.add_argument("--resize_height", default=320, type=int)
    args = parser.parse_args()

    dataset = KITTIHorizonRawMonodepth(dataset_path=args.path, resize_height=args.resize_height, resize_width=args.resize_width)
    cam_numbers = (2, 3)

    if args.whole_data:
        for date_id in tqdm(dataset.get_date_ids()):
            for drive_id in tqdm(dataset.get_drive_ids(date_id)):
                drive = dataset.get_drive(date_id, drive_id)
                for camera in range(2, 4):
                    num_images = len(drive)
                    for idx, images in tqdm(enumerate(iter(drive.rgb))):
                        for cam_number in cam_numbers:
                            cam_idx = 0 if cam_number == 2 else 1
                            print(f"drive: {drive_id}, idx: {idx}, camera: {cam_number}")
                            breakpoint()
                            data = dataset.process_single_image(drive, images[cam_idx], idx, args.camera)
                            processed_image = np.transpose(data["image"], [1, 2, 0])
                            hp1 = data["horizon_p1"]
                            hp2 = data["horizon_p2"]

                            # save_path = Path("/home/gkinoshita/workspace/monodepth2/kitti_data/")

                            # with open(data["save_path"], "w") as f:
                            #     f.write(f"{hp1[0]} {hp1[1]} {hp2[0]} {hp2[1]}")  # x1, y1, x2, y2
                            plt.figure(figsize=(20, 5))
                            plt.imshow(processed_image)
                            plt.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], "r-", lw=2)
                            plt.tight_layout()
                            plt.axis("off")
                            plt.show()

    else:
        if args.date is None:
            all_dates = dataset.get_date_ids()
            print("available dates:")
            for date in all_dates:
                print(date)
            print("specify via the --date option")
            exit(0)

        if args.drive is None:
            all_drives = dataset.get_drive_ids(args.date)
            print("available drives:")
            for drive in all_drives:
                print(drive)
            print("specify via the --drive option")
            exit(0)

        drive = dataset.get_drive(args.date, args.drive)

        num_images = len(drive)

        for idx, image in tqdm(enumerate(iter(drive.rgb))):
            data = dataset.process_single_image(drive, image, idx, args.camera)
            processed_image = np.transpose(data["image"], [1, 2, 0])
            hp1 = data["horizon_p1"]
            hp2 = data["horizon_p2"]

            # with open(data["save_path"], "w") as f:
            #     # f.write(" ".join([str(elem) for elem in [*hp1.tolist(), *hp2.tolist()]]))
            #     f.write(f"{hp1[0]} {hp1[1]} {hp2[0]} {hp2[1]}")  # x1, y1, x2, y2

            plt.figure(figsize=(20, 5))
            plt.imshow(processed_image)
            plt.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], "r-", lw=2)
            plt.tight_layout()
            plt.axis("off")
            plt.show()
