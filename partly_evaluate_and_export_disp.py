import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import datasets
import networks
from eval_utils import batch_post_process_disparity, compute_errors, search_last_epoch
from layers import disp_to_depth
from options import MonodepthOptions

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

segms_labels_str_set = ("segms", "labels")


def collate_fn(dict_batch):
    # batch: [{...} x batch_size]
    ret_dct = {key: default_collate([d[key] for d in dict_batch]) for key in dict_batch[0] if key not in segms_labels_str_set}
    padded_segms, padded_labels, n_insts = pad_segms_labels([d["segms"] for d in dict_batch], [d["labels"] for d in dict_batch])
    ret_dct["padded_segms"] = padded_segms
    ret_dct["padded_labels"] = padded_labels
    ret_dct["n_insts"] = n_insts
    return ret_dct


def pad_segms_labels(batch_segms, batch_labels):
    n_insts = torch.tensor([len(item) for item in batch_labels])
    padded_batch_segms = rnn.pad_sequence(batch_segms, batch_first=True, padding_value=0)
    padded_batch_labels = rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0)
    return padded_batch_segms, padded_batch_labels, n_insts


def evaluate_and_export_split_disp(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    MAX_N_INST = 4

    log_path = (
        Path(opt.root_log_dir)
        / f"{opt.width}x{opt.height}"
        / f"{opt.model_name}{'_' if opt.model_name and opt.ckpt_timestamp else ''}{opt.ckpt_timestamp}"
    )
    save_dir: Path = log_path / "result"
    print(f"All results are saved at {save_dir}")

    pred_disps_split = [[] for _ in range(MAX_N_INST + 1)]

    models_dir = log_path / "models"
    opt.epoch_for_eval = search_last_epoch(models_dir) if opt.epoch_for_eval is None else opt.epoch_for_eval
    weights_dir = models_dir / f"weights_{opt.epoch_for_eval}"

    if not opt.enable_loading_disp_to_eval:
        save_dir.mkdir(parents=False, exist_ok=True)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        with open(os.path.join(splits_dir, opt.eval_split, "test_files.txt")) as f:
            test_filenames = f.readlines()

        print("-> Loading model from ", weights_dir)
        encoder_path = weights_dir / "encoder.pth"
        depth_decoder_path = weights_dir / "depth.pth"

        # LOADING PRETRAINED MODEL
        print("Loading pretrained encoder")
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

        dataset = datasets.KITTIRAWDatasetWithSegm(
            opt.data_path, test_filenames, feed_height, feed_width, [0], 4, is_train=False, segm_dirname=opt.segm_dirname
        )
        dataloader = DataLoader(
            dataset,
            16,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        n_insts = []
        pred_disps = []

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for data in tqdm(dataloader, dynamic_ncols=True):
                input_color = data[("color", 0, 0)].to(device)
                n_inst = data["n_insts"]  # Size([16 (== batch_size)])
                n_inst[n_inst > MAX_N_INST] = MAX_N_INST  # MAX_N_INST個以上のインスタンスが登場するdispはまとめる
                n_inst = n_inst.to(torch.uint8)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output_dict = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output_dict[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                n_insts.append(n_inst)
        pred_disps = np.concatenate(pred_disps)
        n_insts = np.concatenate(n_insts)
        for split in range(MAX_N_INST + 1):
            pred_disps_split[split] = pred_disps[n_insts == split]

        for split in range(MAX_N_INST + 1):
            if split == MAX_N_INST:
                disp_output_path = save_dir / f"disps_{opt.eval_split}_split_{opt.epoch_for_eval}_more_inst.npy"
            else:
                disp_output_path = save_dir / f"disps_{opt.eval_split}_split_{opt.epoch_for_eval}_{split}_inst.npy"
            print("-> Saving predicted disps to", disp_output_path)
            np.save(disp_output_path, pred_disps_split[split])
        np.save(save_dir / "n_insts.npy", n_insts)

    else:
        n_insts = np.load(save_dir / "n_insts.npy")
        MAX_N_INST = n_insts.max()
        for split in range(MAX_N_INST + 1):
            if split == MAX_N_INST:
                disp_output_path = save_dir / f"disps_{opt.eval_split}_split_{opt.epoch_for_eval}_more_inst.npy"
            else:
                disp_output_path = save_dir / f"disps_{opt.eval_split}_split_{opt.epoch_for_eval}_{split}_inst.npy"
            print("-> Saving predicted disps to", disp_output_path)
            if not Path(disp_output_path).exists():
                raise FileNotFoundError(f"{disp_output_path} does not exists. Please remove --enable_loading_disp_to_eval")
            print(f"\n-> Loading predictions from {disp_output_path}")
            pred_disps_split[split] = np.load(disp_output_path)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(splits_dir / "benchmark" / "eigen_to_benchmark_ids.npy")
            for split in range(MAX_N_INST + 1):
                pred_disps_split[split] = pred_disps_split[split][eigen_to_benchmark_ids]

    gt_path: Path = Path(splits_dir) / opt.eval_split / "gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    gt_depths_split = [[] for _ in range(MAX_N_INST + 1)]
    for i, n_inst in enumerate(n_insts):
        gt_depths_split[n_inst].append(gt_depths[i])

    print("-> Evaluating")

    errors_split = [[] for _ in range(MAX_N_INST + 1)]

    for split in range(MAX_N_INST + 1):
        pred_disps = pred_disps_split[split]
        gt_depths = gt_depths_split[split]

        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= opt.pred_depth_scale_factor

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors_split[split].append(compute_errors(gt_depth, pred_depth))

    output_filename = f"partly_result_{opt.epoch_for_eval}.txt"
    mean_errors_split = np.array([np.array(errors_split[split]).mean(0) for split in range(MAX_N_INST + 1)])
    n_n_insts = np.array([(n_insts == split).sum() for split in range(MAX_N_INST + 1)])
    total_mean_errors = (mean_errors_split * n_n_insts[:, None]).sum(0) / n_n_insts.sum()
    with open(save_dir / output_filename, "w") as f:
        print("\t\t" + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        f.write("|          |" + ("{:>8} | " * 8).format("abse", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\n")
        f.write("| - " * 9 + "|\n")
        for split in range(MAX_N_INST + 1):
            mean_errors = mean_errors_split[split]

            if split == MAX_N_INST:
                print(f"{split}=< insts:\t", end="")
                f.write(f"|{split}=< insts:")
            else:
                print(f"{split} insts:\t", end="")
                f.write(f"|{split} insts:  ")
            print(("|{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "|")
            f.write(("|{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "|\n")

        print("total:\t\t", end="")
        print(("|{: 8.3f}  " * 8).format(*total_mean_errors.tolist()) + "|")
        f.write("|total:    ")
        f.write(("|{: 8.3f}  " * 8).format(*total_mean_errors.tolist()) + "|\n")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate_and_export_split_disp(options.parse())
