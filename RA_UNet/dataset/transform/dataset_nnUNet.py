import math

import torch
import torch.utils.data as data
import numpy as np
import os, sys
from dataset.transform.transform import get_training_transforms
import pickle

class VolumeDataset_nn(data.Dataset):
    def __init__(self,
                 input_dir=None
                 ):
        super(VolumeDataset_nn, self).__init__()
        self.input_dir = input_dir
        self.rimg_files = []
        self.mask_files = []
        self.rimg_origin_shape = []
        for x in os.listdir(input_dir):
            if x.endswith('_seg.npy'):
                self.mask_files.append(x)
            elif x.endswith('.pkl'):
                file_path = os.path.join(input_dir, x)
                # 打开 .pkl 文件
                file = open(file_path, 'rb')
                # 使用 pickle.load() 函数读取文件内容
                data = pickle.load(file)
                self.rimg_origin_shape.append(data['shape_before_cropping'])
            elif x.endswith('.npz'):
                continue
            else:
                self.rimg_files.append(x)

        self.cur_rimg_nii = None
        self.cur_cimg_nii = None
        self.cur_bmsk_nii = None

    def getCurRimgNii(self):
        return self.cur_rimg_nii

    def getCurCimgNii(self):
        return self.cur_cimg_nii

    def getCurBmskNii(self):
        return self.cur_bmsk_nii

    def __len__(self):
        return len(self.rimg_files)

    def __getitem__(self, index):
        Out = list()
        if isinstance(self.rimg_files, list):
            rimg_npy = np.load(os.path.join(self.input_dir, self.rimg_files[index]))
            Out.append(rimg_npy)
            self.cur_rimg_nii = rimg_npy

        if isinstance(self.mask_files, list):
            bmsk_npy = np.load(os.path.join(self.input_dir, self.mask_files[index]))
            Out.append(bmsk_npy)
            self.cur_bmsk_nii = bmsk_npy

        if len(Out) == 1:
            Out = Out[0]
        else:
            Out = tuple(Out)
        return (Out, self.rimg_origin_shape[index])


class BlockDataset_nn(data.Dataset):
    def __init__(self,
                rimg=None,
                bmsk=None,
                origin_shape=None):
        super(BlockDataset_nn, self).__init__()

        if isinstance(bmsk, torch.Tensor) and rimg.shape != bmsk.shape:
            print("Invalid shape of image")
            return
        self.raw_shape = rimg.shape
        self.max_dim = torch.tensor(self.raw_shape).max()
        # self.rimg = torch.from_numpy(rimg)
        # self.bmsk = torch.from_numpy(bmsk)
        self.rimg = rimg
        self.bmsk = bmsk
        self.num_slice = 1
        self.transform = None
        self.origin_shape = origin_shape
        self.len = self.raw_shape[0]+self.raw_shape[1]+self.raw_shape[2]
        self.patch_size = [602,602]
        self.work_index = None

        self.transforms = get_training_transforms(
            patch_size=[512, 512],
            rotation_for_DA=(-3.141592653589793, 3.141592653589793),
            deep_supervision_scales=[[1.0, 1.0]],
            mirror_axes=(0, 1),
            do_dummy_2d_data_aug=False,
            use_mask_for_norm=[False],
            is_cascaded=False, foreground_labels=[1, 2],
            regions=None,
            ignore_label=None)

    def getOriginShape(self):
        return self.origin_shape

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.work_index = index
        if index < self.raw_shape[0]:
            rimg_tmp = self.rimg[range(index, index+1), :, :]
            bmsk_tmp = self.bmsk[range(index, index+1), :, :]
        elif index < self.raw_shape[0] + self.raw_shape[1]:
            sind = index - self.raw_shape[0]
            rimg_tmp = self.rimg[:, range(sind, sind+1), :]
            rimg_tmp = rimg_tmp.permute([1, 0, 2])
            bmsk_tmp = self.bmsk[:, range(sind, sind+1), :]
            bmsk_tmp = bmsk_tmp.permute([1, 0, 2])
        else:
            sind = index - self.raw_shape[0] - self.raw_shape[1]
            rimg_tmp = self.rimg[:, :, range(sind, sind+1)]
            rimg_tmp = rimg_tmp.permute([2, 0, 1])
            bmsk_tmp = self.bmsk[:, :, range(sind, sind+1)]
            bmsk_tmp = bmsk_tmp.permute([2, 0, 1])


        rimg_blk = torch.zeros([self.num_slice, self.max_dim, self.max_dim], dtype=torch.float32)
        rimg_blk[:, :rimg_tmp.shape[1], :rimg_tmp.shape[2]] = rimg_tmp
        bmsk_blk = torch.zeros([self.num_slice, self.max_dim, self.max_dim], dtype=torch.long)
        bmsk_blk[:, :bmsk_tmp.shape[1], :bmsk_tmp.shape[2]] = bmsk_tmp

        # transform
        x_1 = int((self.patch_size[0] - rimg_blk.shape[1])/2)
        x_2 = self.patch_size[0] - rimg_blk.shape[1] - x_1
        y_1 = int((self.patch_size[1] - bmsk_blk.shape[2])/2)
        y_2 = self.patch_size[1] - bmsk_blk.shape[2] - y_1
        padding = [[x_1, x_2], [y_1, y_2]]
        rimg_r = np.pad(rimg_blk, ((0, 0), *padding), 'constant', constant_values=0)
        bmsk_r = np.pad(bmsk_blk, ((0, 0), *padding), 'constant', constant_values=-1)

        re = self.transforms(**{'image': torch.from_numpy(rimg_r), 'segmentation': torch.from_numpy(bmsk_r)})

        return re['image'], re['segmentation'][0]

class BlockDataset_nn_v(data.Dataset):
    def __init__(self,
                rimg=None,
                bmsk=None,
                origin_shape=None):
        super(BlockDataset_nn_v, self).__init__()

        if isinstance(bmsk, torch.Tensor) and rimg.shape != bmsk.shape:
            print("Invalid shape of image")
            return

        self.max_dim = torch.tensor(rimg.shape).max()
        # self.rimg = torch.from_numpy(rimg)
        # self.bmsk = torch.from_numpy(bmsk)

        mid = math.floor(self.max_dim / 2)
        mid_down = mid - 256
        mid_up = mid + 256

        self.rimg = rimg[:, mid_down:mid_up, mid_down:mid_up]
        self.bmsk = bmsk[:, mid_down:mid_up, mid_down:mid_up]
        self.num_slice = 1
        self.transform = None
        self.origin_shape = origin_shape
        self.len = origin_shape[0]+origin_shape[1]+origin_shape[2]
        self.work_index = None

        self.transforms = get_training_transforms(
            patch_size=[512, 512],
            rotation_for_DA=(-3.141592653589793, 3.141592653589793),
            deep_supervision_scales=[[1.0, 1.0]],
            mirror_axes=(0, 1),
            do_dummy_2d_data_aug=False,
            use_mask_for_norm=[False],
            is_cascaded=False, foreground_labels=[1, 2],
            regions=None,
            ignore_label=None)

    def getOriginShape(self):
        return self.origin_shape

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.work_index = index
        if index < self.origin_shape[0]:
            rimg_tmp = self.rimg[range(index, index+1), :, :]
            bmsk_tmp = self.bmsk[range(index, index+1), :, :]
        elif index < self.origin_shape[0] + self.origin_shape[1]:
            sind = index - self.origin_shape[0]
            rimg_tmp = self.rimg[:, range(sind, sind+1), :]
            rimg_tmp = rimg_tmp.permute([1, 0, 2])
            bmsk_tmp = self.bmsk[:, range(sind, sind+1), :]
            bmsk_tmp = bmsk_tmp.permute([1, 0, 2])
        else:
            sind = index - self.origin_shape[0] - self.origin_shape[1]
            rimg_tmp = self.rimg[:, :, range(sind, sind+1)]
            rimg_tmp = rimg_tmp.permute([2, 0, 1])
            bmsk_tmp = self.bmsk[:, :, range(sind, sind+1)]
            bmsk_tmp = bmsk_tmp.permute([2, 0, 1])


        rimg_blk = torch.zeros([self.num_slice, 512, 512], dtype=torch.float32)
        rimg_blk[:, :rimg_tmp.shape[1], :rimg_tmp.shape[2]] = rimg_tmp
        bmsk_blk = torch.zeros([self.num_slice, 512, 512], dtype=torch.long)
        bmsk_blk[:, :bmsk_tmp.shape[1], :bmsk_tmp.shape[2]] = bmsk_tmp

        re = self.transforms(**{'image': rimg_blk, 'segmentation': bmsk_blk})

        return re['image'], re['segmentation'][0]

if __name__ == '__main__':
    input_dir = "E:\PycharmProjects\ResTLU-Net-main\data\\nnUNetPreprocessed"
    dataset = VolumeDataset_nn(input_dir)
    (X,y), origin_shape = dataset.__getitem__(1)
    dataset2 = BlockDataset_nn_v(X[0],y[0], origin_shape)
    for i in range(1000):
        (X, y) = dataset2.__getitem__(i)
        print(X.shape)
        print(y[0].shape)
        print(torch.max(y[0]))
