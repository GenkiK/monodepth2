from pathlib import Path
from shutil import copytree

root_dir = Path("/home/gkinoshita/workspace/monodepth2")
root_disp_dir = root_dir / "output_disp"
root_img_dir = root_dir / "kitti_data"

for date_dir in root_disp_dir.iterdir():
    if not date_dir.is_dir():
        continue
    for scene_dir in date_dir.glob("*sync"):
        # copytree(root_disp_dir / date_dir.name / scene_dir.name / )
        for disp_dir in scene_dir.iterdir():
            copytree(disp_dir, root_img_dir / date_dir.name / scene_dir.name / disp_dir.name.replace("image", "disp"))
