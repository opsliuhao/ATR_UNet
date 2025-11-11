import SimpleITK
import torch
import torch.utils.data as data
import torch.nn as nn
import scipy.io as io
import numpy as np
import nibabel as nib
import os, sys


class VolumeDataset(data.Dataset):
    def __init__(self, raw_path, label_path, rescale_size=256):
        super(VolumeDataset, self).__init__()
        self.raw_path = raw_path
        self.label_path = label_path
        self.raw_files = os.listdir(self.raw_path)
        self.label_files = os.listdir(self.label_path)
        self.rescale_size = rescale_size

    def __len__(self):
        return 2 * len(self.raw_files)

    def __getitem__(self, index):
        raw_nii = SimpleITK.ReadImage(os.path.join(self.raw_path, self.raw_files[index]))
        label_nii = SimpleITK.ReadImage(os.path.join(self.label_path, self.label_files[index]))
        raw_array = SimpleITK.GetArrayFromImage(raw_nii)
        label_array = SimpleITK.GetArrayFromImage(label_nii)

        self.shape = raw_array.shape
        self.rescale = self.rescale_size/max(list(self.shape))

        # 0-1 Normalization
        raw_array = (raw_array - raw_array.min()) / (raw_array.max() - raw_array.min())

        raw_tensor = torch.Tensor(raw_array)
        label_tensor = torch.Tensor(label_array)

        raw_tensor = torch.unsqueeze(raw_tensor, 0)
        raw_tensor = torch.unsqueeze(raw_tensor, 0)
        raw_tensor = nn.functional.interpolate(raw_tensor, scale_factor=0.5, mode="trilinear", align_corners=False)
        self.new_shape = raw_tensor.shape

        label_tensor = torch.unsqueeze(label_tensor, 0)
        label_tensor = torch.unsqueeze(label_tensor, 0)
        label_tensor = nn.functional.interpolate(label_tensor, scale_factor=0.5, mode="nearest")

        new_raw_tensor = torch.zeros(self.rescale_size, self.rescale_size, self.rescale_size)
        new_raw_tensor[:raw_tensor.shape[2], :, :] = raw_tensor

        new_label_tensor = torch.zeros(self.rescale_size, self.rescale_size, self.rescale_size)
        new_label_tensor[:label_tensor.shape[2], :, :] = label_tensor

        return torch.unsqueeze(new_raw_tensor, dim=0), torch.unsqueeze(new_label_tensor, dim=0)

    def getRescale(self):
        return self.rescale

    def getRescaleSize(self):
        return self.rescale_size

    def getRawSize(self):
        return self.shape

    def getNewRawSize(self):
        return self.new_shape





if __name__ == '__main__':
    volume_dataset = VolumeDataset(raw_path="../../data/train/raw", label_path="../../data/train/label")
    X, y = volume_dataset.__getitem__(0)
    print(X.shape, y.shape)
    print(volume_dataset.getRescale())
    print(volume_dataset.getRescaleSize())
    print(volume_dataset.getRawSize())



