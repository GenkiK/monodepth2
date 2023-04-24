from pathlib import Path

from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

height_low = 192
width_low = 640

height_high = 320
width_high = 1024

interp = T.InterpolationMode.LANCZOS
shape_dict = {"high": (height_high, width_high), "low": (height_low, width_low)}

new_dir_name_dict = {"high": "image_1024x320_0{}", "low": "image_640x192_0{}"}
resolutions = tuple(shape_dict.keys())
resizer_dict = {k: T.Resize(v, interpolation=interp) for k, v in shape_dict.items()}


root_dir = Path("/home/gkinoshita/workspace/monodepth2/kitti_data/")

for date_dir in tqdm(root_dir.iterdir()):
    for scene_dir in tqdm(date_dir.glob(f"{date_dir.name}*")):
        for camera_dir in scene_dir.glob("image_*"):
            camera_number = camera_dir.name[-1]
            for resolution in resolutions:
                (scene_dir / new_dir_name_dict[resolution].format(camera_number)).mkdir(parents=True, exist_ok=True)
            for img_path in (camera_dir / "data").glob("*.jpg"):
                img = Image.open(img_path)
                for resolution in resolutions:
                    resized_img = resizer_dict[resolution](img)
                    resized_img.save(scene_dir / new_dir_name_dict[resolution].format(camera_number) / img_path.name)
