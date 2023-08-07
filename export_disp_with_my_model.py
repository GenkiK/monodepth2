import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
from tqdm import tqdm

import networks
from eval_utils import search_last_epoch
from layers import disp_to_depth

CAM_NUMBER = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Simple testing function for Monodepthv2 models.")

    parser.add_argument("--root_dir", type=Path, help="path to image director", required=True)
    parser.add_argument("--resolution", type=str, help="input image resolution", choices=["1024x320", "640x192"])
    parser.add_argument("--ext", type=str, help="image extension to search for in folder", default="jpg")
    parser.add_argument("--no_cuda", help="if set, disables CUDA", action="store_true")
    parser.add_argument("--epoch_for_eval", help="the number epochs for using evaluation", type=int)
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--ckpt_timestamp", help="timestamp of the model checkpoint", type=str)
    return parser.parse_args()


def export_disp(args):
    """Function to predict for a single image or folder of images"""
    width, height = map(int, args.resolution.split("x"))
    log_path: Path = (
        args.root_dir
        / "new_logs"
        / args.resolution
        / f"{args.model_name}{'_' if args.model_name and args.ckpt_timestamp else ''}{args.ckpt_timestamp}"
    )
    models_dir = log_path / "models"
    args.epoch_for_eval = search_last_epoch(models_dir) if args.epoch_for_eval is None else args.epoch_for_eval
    weights_dir = models_dir / f"weights_{args.epoch_for_eval}"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", weights_dir)
    encoder_path = weights_dir / "encoder.pth"
    decoder_path = weights_dir / "depth.pth"

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict_dec = torch.load(decoder_path, map_location=device)
    decoder.load_state_dict(loaded_dict_dec)

    decoder.to(device)
    decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        root_img_dir = args.root_dir / "kitti_data"
        for date_dir in root_img_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for scene_dir in date_dir.glob("*sync"):
                for img_dir in scene_dir.glob(f"image_{args.resolution}_0{CAM_NUMBER}"):
                    output_dir = scene_dir / f"disp_{args.resolution}_{args.model_name}_0{CAM_NUMBER}"
                    output_dir.mkdir(parents=False, exist_ok=True)
                    for img_path in tqdm(sorted(img_dir.glob(f"*.{args.ext}"))):
                        # Load image and preprocess
                        input_image = pil.open(img_path).convert("RGB")
                        original_width, original_height = input_image.size
                        input_image = input_image.resize((width, height), pil.LANCZOS)
                        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                        # PREDICTION
                        input_image = input_image.to(device)
                        outputs = decoder(encoder(input_image))

                        disp = outputs[("disp", 0)]
                        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

                        # Saving numpy file
                        scaled_disp, _ = disp_to_depth(disp)
                        name_dest_npy = output_dir / f"{img_path.stem}.npy"
                        np.save(name_dest_npy, scaled_disp.cpu().numpy())

                        # Saving colormapped depth image
                        disp_resized_np = disp_resized.squeeze().cpu().numpy()
                        vmax = np.percentile(disp_resized_np, 95)
                        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
                        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                        im = pil.fromarray(colormapped_im)

                        name_dest_im = output_dir / f"{img_path.stem}.jpg"
                        im.save(name_dest_im)
                        print("   - {}".format(name_dest_im))
                        print("   - {}".format(name_dest_npy))

                print("-> Done!")


if __name__ == "__main__":
    args = parse_args()
    export_disp(args)
