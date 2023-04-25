from pathlib import Path

import numpy as np

from kitti_horizon.kitti_horizon_raw_monodepth import KITTIHorizonRawMonodepth


def horizon2edge_pos(horizon: np.ndarray, h: int, w: int) -> np.ndarray:
    a, b, c = horizon
    xs = np.array((1, w - 1))
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


if __name__ == "__main__":
    for h, w in ((320, 1024), (192, 640)):
        camera_numbers = ("2", "3")
        K = np.array([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]], dtype=np.float32)
        K[0, :] *= w
        K[1, :] *= h
        resolution = f"{w}x{h}"

        RAW_ROOT_DIR = Path("/home/gkinoshita/dataset/packnet-kitti-raw/KITTI_raw")
        ROOT_DIR = Path("/home/gkinoshita/workspace/monodepth2/kitti_data/")

        dataset = KITTIHorizonRawMonodepth(dataset_path=RAW_ROOT_DIR, resize_height=h, resize_width=w)
        for date_dir in ROOT_DIR.iterdir():
            if not date_dir.is_dir():
                continue
            for drive_dir in sorted(date_dir.glob("*sync")):
                drive_id = drive_dir.name[-9:-5]
                drive = dataset.get_drive(date_dir.name, drive_id)

                for cam_number in camera_numbers:
                    data_dir_tpl = str(drive_dir / ("{}_" + resolution + "_0" + cam_number))
                    img_dir = Path(data_dir_tpl.format("image"))
                    save_dir = Path(data_dir_tpl.format("horizon"))
                    save_dir.mkdir(parents=True, exist_ok=True)

                    cam_idx = 0 if cam_number == "2" else 1

                    for img_idx, img_path in enumerate(sorted(img_dir.glob("*jpg"))):
                        data = dataset.process_single_image(drive, drive.get_rgb(img_idx)[cam_idx], img_idx, int(cam_number))
                        processed_img = data["image"].transpose(1, 2, 0)
                        hp1 = data["horizon_p1"]
                        hp2 = data["horizon_p2"]

                        xs, ys = horizon2edge_pos(pos2horizon(hp1[0], hp1[1], hp2[0], hp2[1]), h, w)
                        save_path = save_dir / f"{img_path.stem}.npy"
                        np.save(save_path, np.concatenate((xs, ys), axis=0))

                        # plt.figure(figsize=(20, 5))
                        # plt.imshow(processed_img)
                        # plt.plot(xs_gt, ys_gt, "r-", lw=3, label="GT")
                        # plt.legend()
                        # plt.axis("off")
                        # plt.tight_layout()
                        # plt.show()
                        # plt.close()
