import torch
import torch.nn as nn
import scipy.ndimage as snd
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as PIL
from dataset.transform.dataset_nnUNet import VolumeDataset_nn,BlockDataset_nn_v
from torch.utils.data import DataLoader
import os, sys
import nibabel as nib
import argparse


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        self.exit(2)


def write_nifti(data, aff, shape, out_path):
    data = data[0:shape[0], 0:shape[1], 0:shape[2]]
    img = nib.Nifti1Image(data, aff)
    img.to_filename(out_path)


def rotate_volume(vol):
    tp_trans = ToPILImage()
    tt_trans = ToTensor()

    angle = np.array([1, 1, 1])
    for i in range(3):
        if i == 0:
            old_vol = vol
        dim = old_vol.shape[i]
        for j in range(dim):
            if i == 0:
                one_slice = old_vol[j, :, :]
            elif i == 1:
                one_slice = old_vol[:, j, :]
            else:  # i==2
                one_slice = old_vol[:, :, j]
            one_slice_pil = tp_trans(one_slice)
            one_slice_pil = PIL.rotate(one_slice_pil, angle[i],
                                       resample=PIL.Image.BILINEAR, expand=True)
            one_slice = tt_trans(one_slice_pil)
            if j == 0: pass  # Create New Vol


import numpy as np


def estimate_dice(gt_msk, prt_msk):
    dice_1 = 0
    dice_2 = 0
    # 处理标签为1的情况
    gt_1 = np.where(gt_msk == 1, 1, 0)
    prt_1 = np.where(prt_msk == 1, 1, 0)
    intersection_1 = gt_1 * prt_1
    if gt_1.sum() + prt_1.sum() > 0:
        dice_1 = 2 * float(intersection_1.sum()) / float(gt_1.sum() + prt_1.sum())

    # 处理标签为2的情况
    gt_2 = np.where(gt_msk == 2, 1, 0)
    prt_2 = np.where(prt_msk == 2, 1, 0)
    intersection_2 = gt_2 * prt_2
    if gt_2.sum() + prt_2.sum() > 0:
        dice_2 = 2 * float(intersection_2.sum()) / float(gt_2.sum() + prt_2.sum())

    return dice_1, dice_2


def extract_large_comp(prt_msk):
    labs, num_lab = snd.label(prt_msk)
    c_size = np.bincount(labs.reshape(-1))
    c_size[0] = 0
    max_ind = c_size.argmax()
    prt_msk = labs == max_ind

    return prt_msk


def predict_volumes(model, input_dir,
                    save_dice=False):
    use_gpu = torch.cuda.is_available()
    model_on_gpu = next(model.parameters()).is_cuda
    use_bn = True
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

    volume_dataset = VolumeDataset_nn(input_dir=input_dir)
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1)
    with torch.no_grad():
        for idx, ((X,y),origin_shape) in enumerate(volume_loader):
            ptype = 1  # Predict
            block_dataset = BlockDataset_nn_v(X[0][0],y[0][0],origin_shape)
            # block_loader = DataLoader(dataset=block_dataset, batch_size=1, shuffle=False)
            raw_shape = origin_shape

            r_1 = torch.zeros((3,512,512,512))
            r_2 = torch.zeros((3,512,512,512))
            r_3 = torch.zeros((3,512,512,512))
            m_1 = torch.zeros(origin_shape)

            for i in range(block_dataset.__len__()):
                print(i)
                X, y = block_dataset.__getitem__(i)
                if i < origin_shape[0]:
                    # pre_temp = model(torch.unsqueeze(Variable(X), 0))
                    # r_1[:,i,:,:] = pre_temp.squeeze(0)
                    # m_1[i,:,:] = y
                    continue
                elif i < origin_shape[0]+origin_shape[1]:
                    # pre_temp = model(torch.unsqueeze(Variable(X), 0))
                    # r_2[:,:,i-origin_shape[0],:] = pre_temp.squeeze(0).unsqueeze(2)
                    continue
                else:
                    # r_3[:,:,:,i-origin_shape[0]-origin_shape[1]] = model(torch.unsqueeze(Variable(X), 0)).squeeze(0).unsqueeze(3)
                    continue
            rr = torch.stack((r_1,r_2,r_3), 0)
            rr = torch.mean(rr,0)
            rr = rr[:,:origin_shape[0],:,:]
            rr = torch.argmax(rr, dim=0)
            rr = extract_large_comp(rr)
            dice1, dice2 = estimate_dice(m_1, rr)

            if save_dice:
                dice_dict["dice1"].append(dice1)
                dice_dict["dice2"].append(dice2)

        if save_dice:
            return dice_dict
