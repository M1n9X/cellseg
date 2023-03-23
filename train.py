import os
import time
from argparse import ArgumentParser
from math import ceil

import numpy as np
import torch
from loss.hovernet_loss import HoVerNetLoss
from ml.hovernet import HoVerNet
from monai.data import DataLoader, Dataset, PILReader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (Activations, AsDiscrete, AsDiscreted,
                              BoundingRect, CastToTyped, CenterSpatialCropd,
                              Compose, ComputeHoVerMapsd, EnsureChannelFirstd,
                              FillHoles, Lambdad, LoadImaged, RandFlipd,
                              RandGaussianSmoothd, RandRotate90d,
                              RandSpatialCropd, ResizeWithPadOrCropd,
                              ScaleIntensityRanged, SplitDimd)
from monai.utils import convert_to_tensor, set_determinism
from monai.utils.enums import HoVerNetBranch

from transforms import (GenerateDistanceMapd, GenerateInstanceBorderd,
                        GenerateWatershedMarkersd, GenerateWatershedMaskd,
                        ResizeWithPadOrRandCropd, Watershedd)


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


def run(fold, args):
    max_epochs = args.max_epochs
    val_interval = args.val_freq
    best_metric = -1
    best_metric_epoch = -1
    metric_values = []

    model_path = "/path/to/model"
    data_path = "/path/to/data"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = HoVerNet(n_classes=None).to(device)
    loss_function = HoVerNetLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    os.makedirs(model_path, exist_ok=True)
    img_path = os.path.join(data_path, "images")
    gt_path = os.path.join(data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split(
        ".")[0] + "_label.png" for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.1
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_files = [
        {"image": os.path.join(img_path, img_names[i]),
         "label": os.path.join(gt_path, gt_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"image": os.path.join(img_path, img_names[i]),
         "label": os.path.join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )

    train_transforms = Compose(
        [
            LoadImaged(
                keys=["image", "label"], reader=PILReader, dtype=np.uint8
            ),
            EnsureChannelFirstd(keys=("image", "label"), channel_dim=-1),
            # RandSpatialCropd(
            #     keys=["image", "label"], roi_size=(256, 256), random_size=False
            # ),
            # ResizeWithPadOrCropd(
            #     keys=["image", "label"], spatial_size=(256, 256)),
            ResizeWithPadOrRandCropd(
                keys=["image", "label"], spatial_size=(256, 256)),
            ComputeHoVerMapsd(keys="label"),
            CastToTyped(keys=["image", "label", "hover_label"],
                        dtype=torch.float32),
            AsDiscreted(keys=["label"], to_onehot=2),
            # CenterSpatialCropd(
            #     keys=["label", "hover_label"], roi_size=(164, 164)),

            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=["image", "label", "hover_label"],
                      prob=0.5, spatial_axis=0),
            RandRotate90d(
                keys=["image", "label", "hover_label"], prob=0.5, max_k=1),
            RandGaussianSmoothd(keys=["image"], sigma_x=(
                0.5, 1.15), sigma_y=(0.5, 1.15), prob=0.5),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],
                       reader=PILReader, dtype=np.uint8),
            EnsureChannelFirstd(keys=("image", "label"), channel_dim=-1),
            # RandSpatialCropd(
            #     keys=["image", "label"], roi_size=(256, 256), random_size=False
            # ),
            # ResizeWithPadOrCropd(
            #     keys=["image", "label"], spatial_size=(256, 256)),
            ResizeWithPadOrRandCropd(
                keys=["image", "label"], spatial_size=(256, 256)),
            ComputeHoVerMapsd(keys="label"),
            CastToTyped(keys=["image", "label", "hover_label"],
                        dtype=torch.float32),
            AsDiscreted(keys=["label"], to_onehot=2),
            # CenterSpatialCropd(
            #     keys=["label", "hover_label"], roi_size=(164, 164)),
            ScaleIntensityRanged(keys=["image"], a_min=0.0,
                                 a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=12,
                              num_workers=6, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8,
                            num_workers=4, pin_memory=True)

    train_iters = ceil(len(train_ds) / train_loader.batch_size)
    val_iters = ceil(len(val_ds) / val_loader.batch_size)

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"### training for epoch {epoch + 1}/{max_epochs}")
        model.train()
        for idx, batch_data in enumerate(train_loader):

            # if batch_data["image"].shape[2] < 256:
            #     print("image size too samll")

            inputs, label, hover_map = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["hover_label"].to(device),
            )

            labels = {
                HoVerNetBranch.NP: label,
                HoVerNetBranch.HV: hover_map,
            }

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            print(
                f"epoch {epoch+1}/{max_epochs}, step {idx+1}/{train_iters}, train_loss: {loss.item():.4f}")

        lr_scheduler.step()

        if (epoch + 1) % val_interval == 0:
            torch.cuda.empty_cache()
            print(f"### validation for epoch {epoch + 1}/{max_epochs}")
            model.eval()
            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):

                    val_inputs, val_label, val_hover_map = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                        val_data["hover_label"].to(device),
                    )

                    val_labels = {
                        HoVerNetBranch.NP: val_label,
                        HoVerNetBranch.HV: val_hover_map,
                    }

                    val_outputs = model(val_inputs)

                    val_loss = loss_function(val_outputs, val_labels)
                    print(
                        f"step {val_idx+1}/{val_iters}, val_loss: {val_loss.item():.4f}")

                    val_outputs = [post_process(i, device=device)[
                        0] for i in decollate_batch(val_outputs)]

                    val_label = [i for i in decollate_batch(val_label)]

                    dice_metric(y_pred=val_outputs, y=val_label)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                dice_metric.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_path, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
            torch.cuda.empty_cache()
        print(
            f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch} "
        f"total time: {total_time:.4f}")


def main():
    parser = ArgumentParser(description="HoVerNet training torch pipeline")

    parser.add_argument("--n", type=int, default=1,
                        dest="n_fold", help="fold of cross validation")
    parser.add_argument("--bs", type=int, default=8,
                        dest="batch_size", help="batch size")
    parser.add_argument("--ep", type=int, default=300,
                        dest="max_epochs", help="max epochs")
    parser.add_argument("-f", "--val_freq", type=int,
                        default=1, help="validation frequence")

    args = parser.parse_args()

    set_determinism(seed=0)

    for i in range(args.n_fold):
        run(i, args)


if __name__ == "__main__":
    main()
