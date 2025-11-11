import math
import os

import SimpleITK
import torch
from torch import nn
from torch.utils import data


class VolumeDataset(data.Dataset):
    def __init__(self, raw_path, label_path, rescale_size=256):
        super(VolumeDataset, self).__init__()
        self.raw_path = raw_path
        self.label_path = label_path
        self.raw_files = os.listdir(self.raw_path)
        self.label_files = os.listdir(self.label_path)
        self.rescale_size = rescale_size

    def __len__(self):
        return len(self.raw_files)

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
        label_tensor = label_tensor.long()

        return torch.squeeze(raw_tensor, dim=0), torch.squeeze(label_tensor, dim=0)

    def getRescale(self):
        return self.rescale

    def getRescaleSize(self):
        return self.rescale_size

    def getRawSize(self):
        return self.shape

    def getNewRawSize(self):
        return self.new_shape

class BlockDataset(data.Dataset):
    def __init__(self, raw_tensor, label_tensor, rescale_size=256, patch_size=128):
        super(BlockDataset, self).__init__()
        self.raw_tensor = torch.squeeze(raw_tensor, dim=0)
        self.label_tensor = torch.squeeze(label_tensor, dim=0)
        self.rescale_size = rescale_size
        self.patch_size = patch_size
        self.shape = raw_tensor.shape

    def __len__(self):
        return int(self.rescale_size / self.patch_size)**2

    def __getitem__(self, index):
        if index == 0:
            raw_tensor_ = self.raw_tensor[:, :, :128, :128]
            label_tensor = self.label_tensor[:, :, :128, :128]
        elif index == 1:
            raw_tensor_ = self.raw_tensor[:, :, 128:, :128]
            label_tensor = self.label_tensor[:, :, 128:, :128]
        elif index == 2:
            raw_tensor_ = self.raw_tensor[:, :, :128, 128:]
            label_tensor = self.label_tensor[:, :, :128, 128:]
        else:
            raw_tensor_ = self.raw_tensor[:, :, 128:, 128:]
            label_tensor = self.label_tensor[:, :, 128:, 128:]
        final_raw_tensor = torch.zeros((1, 128, 128, 128), dtype=torch.float32)
        final_label_tensor = torch.zeros((1, 128, 128, 128), dtype=torch.int)
        if raw_tensor_.shape[1] >= self.patch_size:
            final_raw_tensor = raw_tensor_[:, :128, :, :]
            final_label_tensor = label_tensor[:, :128, :, :]
        else:
            final_raw_tensor[:, :raw_tensor_.shape[1], :, :] = raw_tensor_[:, :, :, :]
            final_label_tensor[:, :label_tensor.shape[1], :, :] = label_tensor[:, :label_tensor.shape[1], :, :]
        return final_raw_tensor, final_label_tensor

    def get_shape(self):
        return self.shape


if __name__ == '__main__':
    dataset = VolumeDataset(raw_path="../../data/train/raw", label_path="../../data/train/label")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(dataloader):
        print(data[0].shape)
        volume_dataset = BlockDataset(data[0], data[1])
        dataloader = torch.utils.data.DataLoader(volume_dataset, batch_size=1, shuffle=True)
        for i, data_ in enumerate(dataloader):
            print(1, data_[0].shape, data_[1].shape)
