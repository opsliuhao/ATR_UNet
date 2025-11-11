import torch
from torch.utils import data
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.resnet.ResTLU_Attention_Net_model import Re_UNet2d


class TLU_Dataset(data.Dataset):
    # data x,y,z  target list
    def __init__(self, data, target):
        super(TLU_Dataset, self).__init__()
        self.data = data
        self.target = target
        self.num_slice = 3
        self.rescale_shape = self.data.shape
        self.rescale_dim = max(self.rescale_shape)
        slist0 = list()
        for i in range(self.data.shape[0] - self.num_slice + 1):
            slist0.append(range(i, i + self.num_slice))
        self.slist0 = slist0

        slist1 = list()
        for i in range(self.data.shape[1] - self.num_slice + 1):
            slist1.append(range(i, i + self.num_slice))
        self.slist1 = slist1

        slist2 = list()
        for i in range(self.data.shape[2] - self.num_slice + 1):
            slist2.append(range(i, i + self.num_slice))
        self.slist2 = slist2

        self.batch_len = len(self.slist0) + len(self.slist1) + len(self.slist2)

    def get_slist_len_yz(self):
        return len(self.slist0)

    def get_slist_len_xz(self):
        return len(self.slist1)

    def get_slist_len_xy(self):
        return len(self.slist2)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        index = index % self.batch_len
        if index < len(self.slist0):
            sind = self.slist0[index]
            data_s = self.data[sind, :, :]
            if self.target is not None:
                target_s = self.target[sind, :, :]
        elif index < len(self.slist1) + len(self.slist0):
            sind = self.slist1[index - len(self.slist0)]
            data_s = self.data[:, sind, :]
            data_s = data_s.permute([1, 0, 2])
            if self.target is not None:
                target_s = self.target[:, sind, :]
                target_s = target_s.permute([1, 0, 2])
        else:
            sind = self.slist2[index - len(self.slist0) - len(self.slist1)]
            data_s = self.data[:, :, sind]
            data_s = data_s.permute([2, 0, 1])
            if self.target is not None:
                target_s = self.target[:, :, sind]
                target_s = target_s.permute([2, 0, 1])
        slice_shape = data_s.shape
        extend_dim = self.rescale_dim

        data_m = torch.zeros([self.num_slice, extend_dim, extend_dim], dtype=torch.float32)
        data_m[:, :slice_shape[1], :slice_shape[2]] = data_s
        if self.target is not None:
            target_m = torch.zeros([self.num_slice, extend_dim, extend_dim], dtype=torch.int32)
            target_m[:, :slice_shape[1], :slice_shape[2]] = target_s
            targets = [target_m]
            target_ = target_m
            for i in range(4):
                target_ = target_.float()
                target_ = F.interpolate(target_.unsqueeze(0), scale_factor=(0.5, 0.5), mode='nearest').squeeze(0)
                targets.append(target_.to(dtype=torch.int32))
            out = dict()
            out['data'] = data_m
            out['targets'] = targets
            return out
        else:
            out = dict()
            out['data'] = data_m
            return out

#2 1 64 192 160
def predict_3d_1(model,data, device):
    batch_data_list = []
    for batch in range(data.shape[0]):
        temp = data[batch,0,:,:,:]
        temp_ = predict_3d_2(model, temp, device)
        batch_data_list.append(temp_)
    return torch.stack(batch_data_list, dim=0)


def predict_3d_2(model, data, device):
    dataset = TLU_Dataset(data, None)
    data_yz = torch.zeros([3, 64, 192, 160], dtype=torch.float32)
    for sind_index in range(dataset.get_slist_len_yz()):
        x = dataset[sind_index]['data'].unsqueeze(0).to(device, non_blocking=True)
        out_ = model(x)
        del x
        out_ = out_[:,:, :dataset.rescale_shape[1], :dataset.rescale_shape[2]]
        out_ = out_.squeeze(0)
        data_yz[:, sind_index + 1, :, :] = out_
    data_xz = torch.zeros([3, 64, 192, 160], dtype=torch.float32)
    for sind_index in range(dataset.get_slist_len_yz(),dataset.get_slist_len_yz()+dataset.get_slist_len_xz()):
        x = dataset[sind_index]['data'].unsqueeze(0).to(device, non_blocking=True)
        out_ = model(x)
        del x
        out_ = out_[:,:, :dataset.rescale_shape[0], :dataset.rescale_shape[2]]
        # 1 3 64 160
        out_ = out_.squeeze(0)
        data_xz[:,:,sind_index-dataset.get_slist_len_yz()+1,:] = out_
    data_xy = torch.zeros([3, 64, 192, 160], dtype=torch.float32)
    for sind_index in range(dataset.get_slist_len_yz()+dataset.get_slist_len_xz(), dataset.get_slist_len_yz()+dataset.get_slist_len_xz()+dataset.get_slist_len_xy()):
        x = dataset[sind_index]['data'].unsqueeze(0).to(device, non_blocking=True)
        out_ = model(x)
        del x
        out_ = out_[:,:, :dataset.rescale_shape[0], :dataset.rescale_shape[1]]
        out_ = out_.squeeze(0)
        data_xy[:,:,:,sind_index-dataset.get_slist_len_yz()-dataset.get_slist_len_xz()+1] = out_
    cat_data = torch.stack((data_yz, data_xz, data_xy), dim=0)
    out = torch.mean(cat_data, dim=0)
    return out




if __name__ == '__main__':
    X = torch.randn((4,4,4), dtype=torch.float32)
    print(X)
    data = TLU_Dataset(X, None)
    print(data[1])





