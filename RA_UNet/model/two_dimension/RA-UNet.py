
import torch
import torch.nn as nn


def UpConv2dBlock(dim_in, dim_out,
                  kernel_size=4, stride=2, padding=1,
                  bias=True):
    return nn.Sequential(
        nn.ConvTranspose2d(dim_in
                           , dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(0.1)
    )


# feihong
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=1, bias=True):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_out * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or dim_in != BasicBlock.expansion * dim_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim_in, dim_out * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out * BasicBlock.expansion)
            )

    def forward(self, x):  # LeakyReLu ReLu
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


def Conv2dBlock(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, use_bn=True):
    if use_bn:
        return nn.Sequential(
            BasicBlock(dim_in, dim_out, kernel_size, stride, padding, bias),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(0.1)
        )


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class skip_block(nn.Module):

    def __init__(self, d_in, x_in, out_chanel):
        super(skip_block, self).__init__()
        self.upconv = UpConv2dBlock(dim_in=d_in, dim_out=out_chanel)
        self.attention = Attention_block(out_chanel, x_in, out_chanel)
        self.conv = Conv2dBlock(dim_in=out_chanel * 2, dim_out=out_chanel)


    def forward(self, d, x):
        d = self.upconv(d)
        x = self.attention(d, x)
        d = torch.cat([d, x], dim=1)
        d = self.conv(d)
        return d

# feihong
class RA_UNet(nn.Module):
    def __init__(self,
                 dim_in=3,
                 num_class=3,
                 num_conv_block=5,
                 kernel_root=32,
                 deep_supervision=True,
                 use_bn=True):
        super(RA_UNet, self).__init__()

        self.deep_supervision = deep_supervision

        self.layers = dict()
        self.num_conv_block = num_conv_block
        # Conv Layers
        self.conv1 = Conv2dBlock(dim_in, kernel_root, use_bn=use_bn)
        self.conv2 = Conv2dBlock(dim_in=32, dim_out=64, use_bn=use_bn)
        self.conv3 = Conv2dBlock(dim_in=64, dim_out=128, use_bn=use_bn)
        self.conv4 = Conv2dBlock(dim_in=128, dim_out=256, use_bn=use_bn)
        self.conv5 = Conv2dBlock(dim_in=256, dim_out=512, use_bn=use_bn)
        self.conv6 = Conv2dBlock(dim_in=512, dim_out=512, use_bn=use_bn)
        # Upconv Layers
        self.skip1 = skip_block(512, 512, 512)
        self.skip2 = skip_block(512, 256, 256)
        self.skip3 = skip_block(256, 128, 128)
        self.skip4 = skip_block(128, 64, 64)
        self.skip5 = skip_block(64, 32, 32)
        # maxpool
        self.maxpool = nn.MaxPool2d(2)
        # out_layer
        self.out_layer1 = nn.Conv2d(kernel_root, num_class, 1, 1)
        self.out_layer2 = nn.Conv2d(64, num_class, 1, 1)
        self.out_layer3 = nn.Conv2d(128, num_class, 1, 1)
        self.out_layer4 = nn.Conv2d(256, num_class, 1, 1)
        self.out_layer5 = nn.Conv2d(512, num_class, 1, 1)
        # Weight Initialization
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)

    def forward(self, x):
        num_conv_block = self.num_conv_block
        out_list = []
        conv_out = dict()
        conv_out["conv1"] = self.conv1(x)
        temp1 = self.maxpool(conv_out["conv1"])
        conv_out["conv2"] = self.conv2(temp1)
        temp2 = self.maxpool(conv_out["conv2"])
        conv_out["conv3"] = self.conv3(temp2)
        temp3 = self.maxpool(conv_out["conv3"])
        conv_out["conv4"] = self.conv4(temp3)
        temp4 = self.maxpool(conv_out["conv4"])
        conv_out["conv5"] = self.conv5(temp4)
        temp5 = self.maxpool(conv_out["conv5"])
        conv_out["conv6"] = self.conv6(temp5)

        out = self.skip1(conv_out["conv6"], conv_out["conv5"])
        out_list.append(self.out_layer5(out))
        out = self.skip2(out, conv_out['conv4'])
        out_list.append(self.out_layer4(out))
        out = self.skip3(out, conv_out['conv3'])
        out_list.append(self.out_layer3(out))
        out = self.skip4(out, conv_out['conv2'])
        out_list.append(self.out_layer2(out))
        out = self.skip5(out, conv_out['conv1'])
        out_list.append(self.out_layer1(out))

        if self.deep_supervision:
            out_list.reverse()
            return out_list
        else:
            return out_list[-1]

if __name__ == '__main__':
    resTLU_Net = Re_UNet2d(dim_in=1, num_class=3, deep_supervision=True)
    x = torch.randn(1, 1, 512, 512)
    out = resTLU_Net(x)
    for x in out:
        print(x.shape)
