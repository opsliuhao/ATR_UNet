import os
import SimpleITK as sitk
import numpy as np


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


if __name__ == '__main__':
    pre_nii_path = ""
    label_nii_path = ""
    pre_nii_files = os.listdir(pre_nii_path)
    label_nii_files = os.listdir(label_nii_path)
    for (pre_nii_name, label_nii_name) in zip(pre_nii_files, label_nii_files):
        print(pre_nii_name, label_nii_name)
        pre_nii = sitk.ReadImage(os.path.join(pre_nii_path, pre_nii_name))
        label_nii = sitk.ReadImage(os.path.join(label_nii_path, label_nii_name))
        pre_array = sitk.GetArrayFromImage(pre_nii)
        label_array = sitk.GetArrayFromImage(label_nii)
        dice_scores = dice_compute(pre_array, label_array)
        print(f"kidney dice:{dice_scores[1]}  stone dice:{dice_scores[0]}")
