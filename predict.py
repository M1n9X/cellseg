
import argparse
import os
import time
from typing import Tuple, Union

import monai
import numpy as np
import tifffile as tif
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
from monai.transforms import Activations, AsDiscrete, Compose, FillHoles
from monai.utils import convert_to_tensor
from monai.utils.enums import HoVerNetBranch
from skimage import exposure, io, morphology, segmentation

from ml.hovernet import HoVerNet as pl_HoVerNet
from transforms import (GenerateDistanceMapd, GenerateInstanceBorderd,
                        GenerateWatershedMarkersd, GenerateWatershedMaskd,
                        Watershedd)

join = os.path.join

os.sys.path.append(os.getcwd())


def post_process(output, device, return_binary=True):
    post_trans_seg = Compose([
        GenerateWatershedMaskd(keys=HoVerNetBranch.NP.value, softmax=True),
        GenerateInstanceBorderd(
            keys='mask', hover_map_key=HoVerNetBranch.HV, kernel_size=3),
        GenerateDistanceMapd(
            keys='mask', border_key='border', smooth_fn="gaussian"),
        GenerateWatershedMarkersd(keys='mask', border_key='border',
                                  threshold=0.7, radius=2, postprocess_fn=FillHoles()),
        Watershedd(keys='dist', mask_key='mask', markers_key='markers')
    ])
    if HoVerNetBranch.NC.value in output.keys():
        type_pred = Activations(softmax=True)(output[HoVerNetBranch.NC.value])
        type_pred = AsDiscrete(argmax=True)(type_pred)

    pred_inst_dict = post_trans_seg(output)
    pred_inst = pred_inst_dict['dist']

    inst_info_dict = None

    pred_inst = convert_to_tensor(pred_inst, device=device)
    if return_binary:
        pred_inst[pred_inst > 0] = 1
    return (pred_inst, inst_info_dict, pred_inst_dict)


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(
            percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        'Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./input_dir',
                        type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path',
                        default='./out_dir', type=str, help='output path')
    parser.add_argument('--model_path', default='./model_dir',
                        help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False,
                        action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='hovernet',
                        help='select mode: unet, unetr, swinunetr')
    parser.add_argument('--num_class', default=3, type=int,
                        help='segmentation classes')
    parser.add_argument('--input_size', default=256,
                        type=int, help='segmentation classes')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name.lower() == 'hovernet':
        model = pl_HoVerNet(n_classes=None).to(device)

    # checkpoint = torch.load(
    #     join(args.model_path, 'best_Dice_model.pth'), map_location=torch.device(device))
    # model.load_state_dict(checkpoint['model_state_dict'])

    # test postprocess transform
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'best_Dice_model.pth')), map_location=torch.device(device))
    # %%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))

            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(
                    img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:, :, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:, :, i]
                if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
                    pre_img_data[:, :, i] = normalize_channel(
                        img_channel_i, lower=1, upper=99)

            t0 = time.time()
            test_npy01 = pre_img_data/np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(
                0, 3, 1, 2).type(torch.FloatTensor).to(device)
            test_pred_out = sliding_window_inference(
                test_tensor, roi_size, sw_batch_size, model)
            test_pred_out = torch.nn.functional.softmax(
                test_pred_out, dim=1)  # (B, C, H, W)

            test_pred_out = [post_process(i, device=device)[
                0] for i in test_pred_out]

            test_pred_mask = test_pred_out[0][2]["mask"].detach(
            ).cpu().squeeze()

            tif.imwrite(join(output_path, img_name.split('.')[
                        0]+'_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            print(
                f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')


if __name__ == "__main__":
    main()
