#!/usr/bin/env python
import datetime
import json
import time
from fun.function_3d import predict_volumes
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os, sys
import argparse
import numpy as np
from model.three_dimension.SwinUNetr import createSwinUNETR
from dataset.three_dimension.dateset_SwinUETR import VolumeDataset, BlockDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    NoneType = type(None)
    # Argument
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # Required Option
    required.add_argument('-trt1w', '--train_t1w', type=str, default="data/train/raw", help='Train T1w Directory')
    required.add_argument('-trmsk', '--train_msk', type=str, default="data/train/label", help='Train Mask Directory')
    required.add_argument('-out', '--out_dir', type=str, default="data/out", help='Output Directory')
    required.add_argument('-modelType', '--model_type', type=str, default="SwinUNETR", help='Model Type')
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

    if not os.path.exists(args.train_msk) or not os.path.exists(args.train_t1w):
        print("Invalid train directory, please check again!")
        sys.exit(2)

    use_validate = True
    if isinstance(args.validate_msk, NoneType) or isinstance(args.validate_t1w, NoneType) or \
            not os.path.exists(args.validate_msk) or not os.path.exists(args.validate_t1w):
        use_validate = False
        print("NOTE: Do not use validate dataset.")

    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)


    if args.model_type == 'SwinUNETR':
        model = createSwinUNETR((256, 256, 256), 1, 3, feature_size=48)
        model = nn.DataParallel(model)

    if isinstance(args.init_model, str):
        if not os.path.exists(args.init_model):
            print("Invalid init model, please check again!")
            sys.exit(2)
        checkpoint = torch.load(args.init_model, map_location={'cuda:0': 'cpu'})
        model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        model.cuda()
        cudnn.benchmark = True

    # optimizer
    optimizerSs = optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss function
    criterionSs = nn.CrossEntropyLoss()
    if use_gpu:
        criterionSs.cuda()

    dataset = VolumeDataset(args.train_t1w, args.train_msk)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Init Dice and Loss Dict
    DL_Dict = dict()
    dice_list = list()
    loss_list = list()

    # if use_validate:
    #     valid_model = nn.Sequential(model, nn.Softmax())
    #     dice_dict = predict_volumes(valid_model, rimg_in=None, cimg_in=args.validate_t1w, bmsk_in=args.validate_msk,
    #                                 rescale_dim=args.rescale_dim, num_slice=args.input_slice, save_nii=False,
    #                                 save_dice=True)
    #     dice_array = np.array([v for v in dice_dict.values()])
    #     DL_Dict["origin_dice"] = dice_array
    #     print("Origin Dice: %.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))

    DL_Dict['dice1'] = []
    DL_Dict['dice2'] = []
    for epoch in range(0, args.num_epoch):
        lossSs_v = []
        print("Begin Epoch %d" % epoch)
        for i, data in enumerate(dataloader):
            dataset2 = BlockDataset(data[0], data[1])
            dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=True)
            for i2, data_ in enumerate(dataloader2):
                cimg = data_[0].cuda()
                bmsk = data_[1].cuda()
                pr_bmsk = model(cimg)
                # Loss Backward
                lossSs = criterionSs(pr_bmsk, torch.squeeze(bmsk, dim=0).long())
                optimizerSs.zero_grad()
                lossSs.backward()
                optimizerSs.step()

                if use_gpu:
                    lossSs = lossSs.cpu()

                lossSs_v.append(lossSs.data.detach().numpy())
                print(datetime.datetime.now())
                print('\tEpoch:%.2d [%.3d/%.3d][%.3d/%.3d]\tLoss Ss: %.6f'%(epoch, i, len(dataloader.dataset) - 1,i2,len(dataloader2.dataset)-1, lossSs.data.detach()))
        loss = np.array(lossSs_v).sum()
        loss_list.append(loss)

        if use_validate:
            valid_model = nn.Sequential(model, nn.Softmax())
            dice_dict = predict_volumes(valid_model, rimg_in=None, cimg_in=args.validate_t1w, bmsk_in=args.validate_msk,
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

