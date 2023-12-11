# """This is unofficial implementation of YeNet:
# Deep Learning Hierarchical Representation for Image Steganalysis.
# """
# import torch
# from torch import Tensor
# from torch import nn
# import torch.nn.functional as F
# # import sys
# # sys.path.append('./')
# import numpy as np
# import config as c


# class SRMConv(nn.Module):
#     """This class computes convolution of input tensor with 30 SRM filters"""

#     def __init__(self) -> None:
#         """Constructor."""
#         super().__init__()
#         self.device = torch.device(
#             "cuda:0" if torch.cuda.is_available() else "cpu"
#         )
#         self.srm = torch.from_numpy(np.load("models/srm.npy")).to(
#             self.device, dtype=torch.float
#         ).repeat(1, c.stego_img_channel, 1, 1)

#         self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

#     def forward(self, inp: Tensor) -> Tensor:
#         """Returns output tensor after convolution with 30 SRM filters
#         followed by TLU activation."""
#         return self.tlu(F.conv2d(inp, self.srm))


# class ConvBlock(nn.Module):
#     """This class returns building block for YeNet class."""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: int = 0,
#         use_pool: bool = False,
#         pool_size: int = 3,
#         pool_padding: int = 0,
#     ) -> None:
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=True,
#         )
#         self.activation = nn.ReLU()
#         self.pool = nn.AvgPool2d(
#             kernel_size=pool_size, stride=2, padding=pool_padding
#         )
#         self.use_pool = use_pool

#     def forward(self, inp: Tensor) -> Tensor:
#         """Returns conv->gaussian->average pooling."""
#         if self.use_pool:
#             return self.pool(self.activation(self.conv(inp)))
#         return self.activation(self.conv(inp))


# class Model(nn.Module):
#     """This class returns YeNet model."""

#     def __init__(self) -> None:
#         super().__init__()
#         self.layer1 = ConvBlock(30, 30, kernel_size=3)
#         self.layer2 = ConvBlock(30, 30, kernel_size=3)
#         self.layer3 = ConvBlock(
#             30, 30, kernel_size=3, use_pool=True, pool_size=2, pool_padding=0
#         )
#         self.layer4 = ConvBlock(
#             30,
#             32,
#             kernel_size=5,
#             padding=0,
#             use_pool=True,
#             pool_size=3,
#             pool_padding=0,
#         )
#         self.layer5 = ConvBlock(
#             32, 32, kernel_size=5, use_pool=True, pool_padding=0
#         )
#         self.layer6 = ConvBlock(32, 32, kernel_size=5, use_pool=True)
#         self.layer7 = ConvBlock(32, 16, kernel_size=3)
#         # self.layer8 = ConvBlock(16, 16, kernel_size=3, stride=3)
#         self.layer8 = ConvBlock(16, 16, kernel_size=3, stride=3)
#         # self.fully_connected = nn.Sequential(
#         #     nn.Linear(in_features=16 * 3 * 3, out_features=2),
#         #     nn.LogSoftmax(dim=1),
#         # )
#         # self.gap = nn.AdaptiveAvgPool2d(output_size=1) # global average pooling

#         self.fully_connected = nn.Linear(in_features=16 * 8 * 8, out_features=2)


#     def forward(self, image: Tensor) -> Tensor:
#         """Returns logit for the given tensor."""
#         out = SRMConv()(image)
#         out = nn.Sequential(
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.layer4,
#             self.layer5,
#             self.layer6,
#             self.layer7,
#             self.layer8,
#             # self.gap,
#         )(out)
#         out = out.view(out.size(0), -1)
#         out = self.fully_connected(out)
#         return out


# if __name__ == "__main__":
#     net = Model()
#     print(net)
#     inp_image = torch.randn((1, 1, 256, 256))
#     print(net(inp_image))


import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import config as c

SRM_npy = np.load('models/SRM_Kernels.npy')

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = c.stego_img_channel
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1,1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, self.in_channels, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                 stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
                              stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()

class Model(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(Model, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = SRM_conv2d(1, 0)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
        self.block2 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block3 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block4 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 5, with_bn=self.with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(32, 16, 3, with_bn=self.with_bn)
        self.block9 = ConvBlock(16, 16, 3, 3, with_bn=self.with_bn)
        self.num_of_neurons = 144 if c.stego_img_height == 256 else 1024 # (stego_img_height=512)
        # self.ip1 = nn.Linear(3 * 3 * 16, 2)
        self.ip1 = nn.Linear(self.num_of_neurons, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)
        x = self.TLU(x)
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        x = x.view(x.size(0), -1)
        x = self.ip1(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRM_conv2d) or \
                    isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0. ,0.01)
                mod.bias.data.zero_()

def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()