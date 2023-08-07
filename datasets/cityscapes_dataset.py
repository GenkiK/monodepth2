from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def load_pil(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class CityscapesDataset(Dataset):
    """
    Cityscapes dataset for inference
    """

    def __init__(self, data_path: Path, height: int, width: int, orig_height: int, orig_width: int):
        self.data_path = data_path
        self.height = height
        self.width = width
        self.img_paths = sorted(data_path.glob("**/*.png"))
        if orig_height / orig_width < height / width:
            new_h = orig_height
            new_w = int(orig_height * width / height)
        else:
            new_h = int(orig_width * height / width)
            new_w = orig_width
        self.transform = T.Compose(
            [
                T.CenterCrop((new_h, new_w)),
                T.Resize((height, width), interpolation=T.InterpolationMode.LANCZOS),
                T.ToTensor(),
            ]
        )
        self.pil_loader = load_pil

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> torch.Tensor:
        img_path = self.img_paths[index]
        # img = self.pil_loader(img_path)
        img = Image.open(img_path)
        return self.transform(img)
