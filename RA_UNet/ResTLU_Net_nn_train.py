#!/usr/bin/env python
import json
import time
from fun.function_nn import predict_volumes
from dataset.transform.dataset_nnUNet import BlockDataset_nn, VolumeDataset_nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os, sys, pickle
import argparse
import numpy as np
from model.two_dimension import ResTLU_Net_model,SegNet
# from model.two_dimension.VMNet import VMUNet
# from model.two_dimension.deeplab3plus.deeplabv3_plus import DeepLab
# from model.two_dimension.pspnet.pspnet import PSPNet

start = time.time()
device = torch.device("cuda")

if __name__ == '__main__':
    NoneType = type(None)
    # Argument
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # Required Option
    required.add_argument('-trt1w', '--train_t1w', type=str, default="data/train/raw", help='Train T1w Directory')
    required.add_argument('-trmsk', '--train_msk', type=str, default="data/train/label", help='Train Mask Directory')
    required.add_argument('-out', '--out_dir', type=str, default="data/out/nn", help='Output Directory')
    required.add_argument('-modelType', '--model_type', type=str, default="ResTLU_Net", help='Model Type')
    # Optional Option
    optional.add_argument('-vt1w', '--validate_t1w', type=str, default="data/val/raw",help='Validation T1w Directory')
    optional.add_argument('-vmsk', '--validate_msk', type=str, default="data/val/label",help='Validation Mask Directory')
    optional.add_argument('-init', '--init_model', type=str, help='Init Model')
    optional.add_argument('-slice', '--input_slice', type=int, default=3, help='Number of Slice for Model Input')
    optional.add_argument('-conv', '--conv_block', type=int, default=5, help='Number of UNet Block')
    optional.add_argument('-rescale', '--rescale_dim', type=int, default=256, help='Number of the rescale_dim')
    optional.add_argument('-kernel', '--kernel_root', type=int, default=16, help='Number of the Root of Kernel')
    optional.add_argument('-epoch', '--num_epoch', type=int, default=30, help='Number of Epoch')
    optional.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Number of Epoch')
    parser._action_groups.append(optional)

    args = parser.parse_args()

    # if not os.path.exists(args.train_msk) or not os.path.exists(args.train_t1w):
    #     print("Invalid train directory, please check again!")
    #     sys.exit(2)
    #
    use_validate = True
    # if isinstance(args.validate_msk, NoneType) or isinstance(args.validate_t1w, NoneType) or \
    #         not os.path.exists(args.validate_msk) or not os.path.exists(args.validate_t1w):
    #     use_validate = False
    #     print("NOTE: Do not use validate dataset.")

    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)


    if args.model_type == 'ResTLU_Net':
        model = ResTLU_Net_model.Re_UNet2d(dim_in=1, num_conv_block=5, kernel_root=16)
        model = nn.DataParallel(model)
    # if args.model_type == 'SegNet':
    #     model = SegNet.SegNet(3)
    #     model = nn.DataParallel(model)
    # if args.model_type == 'Deeplab3plus':
    #     model = DeepLab(num_classes=3, pretrained=False)
    # if args.model_type == 'PSP_Net':
    #     model = PSPNet(num_classes=3, downsample_factor=8, pretrained=False)
    # if args.model_type == 'VMNet':
    #     model = VMUNet(3, 1)
    if isinstance(args.init_model, str):
        if not os.path.exists(args.init_model):
            print("Invalid init model, please check again!")
            sys.exit(2)
        checkpoint = torch.load(args.init_model, map_location={'cuda:0': 'cpu'})
        model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        model.to(device)
        cudnn.benchmark = True

    # optimizer
    optimizerSs = optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss function
    criterionSs = nn.CrossEntropyLoss()
    if use_gpu:
        criterionSs.cuda()

    volume_dataset = VolumeDataset_nn(input_dir="F:\\nnUNetFrame\\nnUNet_preprocessed\Dataset001_Kidney\\nnUNetPlans_2d\\train")
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # blk_batch_size = 20
    # feihong
    blk_batch_size = 32
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Init Dice and Loss Dict
    DL_Dict = dict()
    dice_list = list()
    loss_list = list()

    # if use_validate:
    #     valid_model = nn.Sequential(model, nn.Softmax2d())
    #     valid_model.eval()
    #     dice_dict = predict_volumes(valid_model, input_dir="F:\\nnUNetFrame\\nnUNet_preprocessed\Dataset001_Kidney\\nnUNetPlans_2d\\val",
    #                                 save_dice=True)
    #     dice_array = np.array([v for v in dice_dict.values()])
    #     DL_Dict["origin_dice"] = dice_array
    #     print("Origin Dice: %.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))

    DL_Dict['dice1'] = []
    DL_Dict['dice2'] = []
    for epoch in range(0, args.num_epoch):
        model.train()
        lossSs_v = []
        print("Begin Epoch %d" % epoch)
        for i, ((X,y), origin_shape) in enumerate(volume_loader):
            block_dataset = BlockDataset_nn(X[0][0], y[0][0], origin_shape)
            block_loader = DataLoader(dataset=block_dataset, batch_size=blk_batch_size, shuffle=True, num_workers=0)
            for j, (cimg_blk, bmsk_blk) in enumerate(block_loader):
                cimg_blk, bmsk_blk = Variable(cimg_blk), Variable(bmsk_blk)
                if use_gpu:
                    cimg_blk = cimg_blk.cuda()
                    bmsk_blk = bmsk_blk.cuda()
                if cimg_blk.shape[0] == 1:
                    continue
                pr_bmsk_blk = model(cimg_blk)

                # Loss Backward
                lossSs = criterionSs(pr_bmsk_blk, bmsk_blk.squeeze(1))
                optimizerSs.zero_grad()
                lossSs.backward()
                optimizerSs.step()

                if use_gpu:
                    lossSs = lossSs.cpu()

                lossSs_v.append(lossSs.data.detach().numpy())

                print('\tEpoch:%.2d [%.3d/%.3d (%.4d/%.4d)]\tLoss Ss: %.6f' % \
                      (epoch, i, len(volume_loader.dataset) - 1,
                       j * blk_batch_size, len(block_loader.dataset),
                       lossSs.data.detach()
                       )
                      )
        loss = np.array(lossSs_v).sum()
        loss_list.append(loss)

        if use_validate:
            model.eval()
            valid_model = nn.Sequential(model, nn.Softmax2d())
            dice_dict = predict_volumes(valid_model,
                                        input_dir="F:\\nnUNetFrame\\nnUNet_preprocessed\Dataset001_Kidney\\nnUNetPlans_2d\\val",
                                        save_dice=True)
            dice_array_1 = np.array(dice_dict['dice1'])
            dice_array_2 = np.array(dice_dict['dice2'])
            DL_Dict['dice1'].append(dice_array_1.mean())
            DL_Dict['dice2'].append(dice_array_2.mean())
            print("\tEpoch: %d; Dice: %.4f +/- %.4f; Loss: %.4f" % (epoch, dice_array_1.mean(), dice_array_1.std(), loss))
        #     feihong
        #     dice_array_mean = dice_array.mean()
        else:
            dice_array = []
            print("\tEpoch: %d; Loss: %.4f" % (epoch, loss))

        if (epoch) % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizerSs': optimizerSs.state_dict(),
                'lossSs': lossSs_v
            }
            torch.save(checkpoint, os.path.join(args.out_dir, 'model-%.2d-epoch' % (epoch)))
    print(DL_Dict)
    file_path = args.out_dir + "/" + args.model_type +'_DL_Dict.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(DL_Dict, f, ensure_ascii=False, indent=4)

end = time.time()
print('running time:%s minutes' % ((end - start) / 60))
