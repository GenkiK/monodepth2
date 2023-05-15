# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
from tqdm import tqdm

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(description="Simple testing function for Monodepthv2 models.")

    parser.add_argument("--root_img_dir", type=Path, help="path to image director", required=True)
    parser.add_argument("--root_output_dir", type=Path, help="root path to output director", required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        help="name of a pretrained model to use",
        choices=[
            "mono",
            "stereo",
            "mono+stereo",
            "mono_no_pt",
            "stereo_no_pt",
            "mono+stereo_no_pt",
        ],
    )
    parser.add_argument("--resolution", type=str, help="input image resolution", choices=["1024x320", "640x192"])
    parser.add_argument("--ext", type=str, help="image extension to search for in folder", default="jpg")
    parser.add_argument("--no_cuda", help="if set, disables CUDA", action="store_true")

    return parser.parse_args()


def export_disp(args):
    """Function to predict for a single image or folder of images"""
    args.model_name = args.model_type + "_" + args.resolution
    assert args.model_name is not None, "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        root_img_dir = Path(args.root_img_dir)
        args.root_output_dir.mkdir(parents=False, exist_ok=True)
        for date_dir in root_img_dir.iterdir():
            if not date_dir.is_dir():
                continue
            output_date_dir = Path(args.root_output_dir) / date_dir.name
            output_date_dir.mkdir(parents=False, exist_ok=True)
            for scene_dir in date_dir.glob("*sync"):
                output_scene_dir = output_date_dir / scene_dir.name
                output_scene_dir.mkdir(parents=False, exist_ok=True)
                for img_dir in scene_dir.glob(f"image_{args.resolution}*"):
                    output_dir = output_scene_dir / img_dir.name
                    output_dir.mkdir(parents=False, exist_ok=True)
                    for img_path in tqdm(sorted(img_dir.glob(f"*.{args.ext}"))):
                        if str(img_path).endswith("_disp.jpg"):
                            # don't try to predict disparity for a disparity image!
                            continue

                        # Load image and preprocess
                        input_image = pil.open(img_path).convert("RGB")
                        original_width, original_height = input_image.size
                        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                        # PREDICTION
                        input_image = input_image.to(device)
                        features = encoder(input_image)
                        outputs = depth_decoder(features)

                        disp = outputs[("disp", 0)]
                        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

                        # Saving numpy file
                        output_name = os.path.splitext(os.path.basename(img_path))[0]
                        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
                        name_dest_npy = output_dir / f"{output_name}_disp.npy"
                        np.save(name_dest_npy, scaled_disp.cpu().numpy())

                        # Saving colormapped depth image
                        disp_resized_np = disp_resized.squeeze().cpu().numpy()
                        vmax = np.percentile(disp_resized_np, 95)
                        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
                        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                        im = pil.fromarray(colormapped_im)

                        name_dest_im = output_dir / f"{output_name}_disp.jpg"
                        im.save(name_dest_im)

                        print("   - {}".format(name_dest_im))
                        print("   - {}".format(name_dest_npy))

                print("-> Done!")


if __name__ == "__main__":
    args = parse_args()
    export_disp(args)
