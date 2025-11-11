import numpy as np
import torch
import os

from torch import nn
from torch.utils.data import DataLoader

from dataset.three_dimension.dateset_SwinUETR import VolumeDataset
from dataset.three_dimension.dateset_SwinUETR import BlockDataset


def dice_compute(pre_array, mask_array):
    """
    计算两个指定标签（标签值为1和2）与背景（标签值为0）之间的Dice系数。

    参数:
    pre_array (numpy.ndarray): 预测的分割结果数组，元素值代表不同的标签类别（0、1、2等）。
    mask_array (numpy.ndarray): 对应的真实掩码数组，元素值代表不同的标签类别（0、1、2等）。

    返回:
    list: 包含两个标签分别对应的Dice系数的列表，顺序与标签值对应（先标签1的Dice系数，后标签2的Dice系数）。
    """
    dice_scores = []
    for label in [1, 2]:
        # 提取预测结果中当前标签对应的像素位置
        pre_mask = (pre_array == label).astype(int)
        # 提取真实掩码中当前标签对应的像素位置
        true_mask = (mask_array == label).astype(int)

        intersection = np.sum(pre_mask * true_mask)
        if intersection == 0:
            dice_score = 0
        else:
            pre_mask_sum = np.sum(pre_mask)
            true_mask_sum = np.sum(true_mask)
            dice_score = (2 * intersection) / (pre_mask_sum + true_mask_sum)

        dice_scores.append(dice_score)

    return dice_scores


def predict_volumes(model, rimg_in=None, cimg_in=None, bmsk_in=None, suffix="pre_mask",
                    save_dice=False, save_nii=False, nii_outdir=None, verbose=False,
                    rescale_dim=256, num_slice=3):
    use_gpu = torch.cuda.is_available()
    model_on_gpu = next(model.parameters()).is_cuda
    if use_gpu:
        if not model_on_gpu:
            model.cuda()
    else:
        if model_on_gpu:
            model.cpu()

    if save_dice:
        dice_dict = dict()
        dice_dict['dice1'] = []
        dice_dict['dice2'] = []

    volume_dataset = VolumeDataset(raw_path=cimg_in, label_path=bmsk_in)
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1)

    for X, y in volume_loader:
        # X (1,256,256,256)
        # y (1,256,256,256)
        rescale_shape = volume_dataset.getRescaleSize()
        raw_shape = volume_dataset.getRawSize()
        block_dateset = BlockDataset(X, y)
        block_loader = DataLoader(dataset=block_dateset, batch_size=1)

        pred_volumes = torch.zeros_like(y)
        pred_volumes = torch.squeeze(pred_volumes,dim=0)
        for i, data in enumerate(block_loader):
            pre = model(data[0])
            pre = torch.argmax(pre, dim=1)
            if y.shape[2] <= 128:
                if i == 0:
                    pred_volumes[:, :, :128, :128] = pre[:, :y.shape[2], :, :]
                elif i == 1:
                    pred_volumes[:, :, 128:, :128] = pre[:, :y.shape[2], :, :]
                elif i == 2:
                    pred_volumes[:, :, :128, 128:] = pre[:, :y.shape[2], :, :]
                else:
                    pred_volumes[:, :, 128:, 128:] = pre[:, :y.shape[2], :, :]
            else:
                if i == 0:
                    pred_volumes[:, :128, :128, :128] = pre[:, :, :, :]
                elif i == 1:
                    pred_volumes[:, :128, 128:, :128] = pre[:, :, :, :]
                elif i == 2:
                    pred_volumes[:, :128, :128, 128:] = pre[:, :, :, :]
                else:
                    pred_volumes[:, :128, 128:, 128:] = pre[:, :, :, :]

        if use_gpu:
            pred_volumes = pred_volumes.cpu()

        # pred_volumes = torch.unsqueeze(pred_volumes, 0)
        # pred_volumes = nn.functional.interpolate(pred_volumes, size=raw_shape, mode="trilinear", align_corners=False)

        pred_volumes = pred_volumes.numpy()

        if isinstance(y, torch.Tensor):
            y = y.data[0].numpy()
            dices = dice_compute(pred_volumes, y)
            if verbose:
                print(dices[0], dices[1])
    #
    #     t1w_nii = volume_dataset.getCurCimgNii()
    #     t1w_path = t1w_nii.get_filename()
    #     t1w_dir, t1w_file = os.path.split(t1w_path)
    #     t1w_name = os.path.splitext(t1w_file)[0]
    #     t1w_name = os.path.splitext(t1w_name)[0]
    #
    #     # if save_nii:
    #     #     t1w_aff = t1w_nii.affine
    #     #     t1w_shape = t1w_nii.shape
    #     #
    #     #     if isinstance(nii_outdir, NoneType):
    #     #         nii_outdir = t1w_dir
    #     #
    #     #     if not os.path.exists(nii_outdir):
    #     #         os.mkdir(nii_outdir)
    #     #     out_path = os.path.join(nii_outdir, t1w_name + "_" + suffix + ".nii.gz")
    #     #     write_nifti(np.array(pr_bmsk_final, dtype=np.float32), t1w_aff, t1w_shape, out_path)
    #
        if save_dice:
            dice_dict["dice1"].append(dices[0])
            dice_dict["dice2"].append(dices[1])
    if save_dice:
        return dice_dict