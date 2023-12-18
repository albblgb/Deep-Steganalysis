import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

import config as c

SRM_npy1 = np.load('models/kernels/SRM3_3.npy')
SRM_npy2 = np.load('models/kernels/SRM5_5.npy')

class pre_Layer_3_3(nn.Module):
    def __init__(self, stride=1, padding=1):
        super(pre_Layer_3_3, self).__init__()
        self.in_channels = c.stego_img_channel
        self.out_channels = 25
        self.kernel_size = (3, 3)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(25, c.stego_img_channel, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy1
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

class pre_Layer_5_5(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(pre_Layer_5_5, self).__init__()
        self.in_channels = c.stego_img_channel
        self.out_channels = 5
        self.kernel_size = (5, 5)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(5, self.in_channels, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(5), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy2
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class spp_layer(nn.Module):

    def __init__(self, batch_size):
        super(spp_layer, self).__init__()

        self.batch_size = batch_size

    def forward(self, x):
        temp = x
        a, b, c, d = x.size()

        x = F.avg_pool2d(x, (c, d), c)

        spp = x.view(self.batch_size * 2, -1)

        x = temp
        x = F.avg_pool2d(x, int(c / 2), int(c / 2))
        spp = torch.cat((spp, x.view(self.batch_size * 2, -1)), 1)

        x = temp
        x = F.avg_pool2d(x, (int(c / 4)), int(c / 4))
        spp = torch.cat((spp, x.view(self.batch_size * 2, -1)), 1)

        return spp


class Basic_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_global):

        super(Basic_Block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU(inplace=True)

        self.use_global = use_global

        self.pool = nn.AvgPool2d(5, 2, 2)

    def forward(self, x):

        if self.use_global:
            x = self.act(self.bn(self.conv(x)))
        else:
            x = self.pool(self.act(self.bn(self.conv(x))))

        return x

class pre_layer(nn.Module):

    def __init__(self):
        super(pre_layer, self).__init__()

        self.conv1 = pre_Layer_3_3()
        self.conv2 = pre_Layer_5_5()


    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.conv2(x)

        return torch.cat((x1, x2),1)  # 1代表在第二个元素进行cat 也就是channel

class conv_Layer(nn.Module):


    def __init__(self):
        super(conv_Layer, self).__init__()

        self.conv1 = nn.Conv2d(30, 60, 3, 1, 1,groups=30)

        self.conv1_1 = nn.Conv2d(60, 30, 1)

        self.bn1 = nn.BatchNorm2d(30)

        self.conv2 = nn.Conv2d(30, 60, 3, 1, 1, groups=30)

        self.conv2_1 = nn.Conv2d(60, 30, 1)

        self.bn2 = nn.BatchNorm2d(30)

        self.conv_layer = nn.Sequential(
            Basic_Block(30, 32, 3, 1, 1, False),
            Basic_Block(32, 32, 3, 1, 1, False),
            Basic_Block(32, 64, 3, 1, 1, False),
            Basic_Block(64, 128, 3, 1, 1, True)
        )

        batch_size = c.train_batch_size if c.mode == 'train' else c.test_batch_size // 2
        self.spp = spp_layer(batch_size)#这里

        self.classfier = nn.Sequential(
            nn.Linear(2688, 1024),
            # nn.Linear(10752, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        temp = x
        x = self.conv1(x).abs()    # abs层，是在第一个卷积操作后进行绝对值操作  提高残差
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x += temp

        x = F.relu(x, inplace=True)

        x = self.conv_layer(x)
        x = self.spp(x)
        # print(x.shape)
        x = self.classfier(x)
        return x


class Zhu_Net(nn.Module):

    def __init__(self):
        super(Zhu_Net, self).__init__()

        self.layer1 = pre_layer()
        self.layer2 = conv_Layer()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
