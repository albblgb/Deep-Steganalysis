import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import config as c


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))


class SRMConv2d(nn.Module):

    def __init__(self, stride=1, padding=0):
        super(SRMConv2d, self).__init__()
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
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, self.in_channels, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class BlockA(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=None):
        super(BlockA, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class BlockB(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=None):
        super(BlockB, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes, stride=2)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        # self.pool = nn.AvgPool2d(3, stride=2, padding=1)

        self.shortcut_conv = conv1x1(in_planes, out_planes, stride=2)
        self.shortcut_bn = norm_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.pool(out)

        identity = self.shortcut_conv(identity)
        identity = self.shortcut_bn(identity)

        out += identity
        out = self.relu(out)

        return out
    

class KeNet(nn.Module):

    def __init__(self, norm_layer=None, zero_init_residual=True, p=0.5):
        super(KeNet, self).__init__()

        self.zero_init_residual = zero_init_residual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.srm = SRMConv2d(1, 0)
        self.bn1 = norm_layer(30)

        self.A1 = BlockA(30, 30, norm_layer=norm_layer)
        self.A2 = BlockA(30, 30, norm_layer=norm_layer)
        self.AA = BlockA(30, 30, norm_layer=norm_layer)

        # self.B1 = BlockB(30, 30, norm_layer=norm_layer)
        # self.B2 = BlockB(30, 64, norm_layer=norm_layer)

        self.B3 = BlockB(30, 64, norm_layer=norm_layer)
        self.A3 = BlockA(64, 64, norm_layer=norm_layer)

        self.B4 = BlockB(64, 128, norm_layer=norm_layer)
        self.A4 = BlockA(128, 128, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.bnfc = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        # self.fcfusion = nn.Linear(128, 128) #4
        self.fc = nn.Linear(128 * 4 + 1, 2)
        self.dropout = nn.Dropout(p=p)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, (BlockA, BlockB)):
                    nn.init.constant_(m.bn2.weight, 0)

    def extract_feat(self, x):
        x = x.float()
        out = self.srm(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.A1(out)
        out = self.A2(out)
        out = self.AA(out)

        # out = self.B1(out)
        # out = self.B2(out)

        out = self.B3(out)
        out = self.A3(out)

        out = self.B4(out)
        out = self.A4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), out.size(1))

        # out = self.relu(out)
        # out = self.bnfc(out)

        return out

    def forward(self, *args):
        ############# statistics fusion start #############
        feats = torch.stack(
            [self.extract_feat(subarea) for subarea in args], dim=0
        )

        euclidean_distance = F.pairwise_distance(feats[0], feats[1], eps=1e-6,
                                                 keepdim=True)

        if feats.shape[0] == 1:
            final_feat = feats.squeeze(dim=0)
        else:
            # feats_sum = feats.sum(dim=0)
            # feats_sub = feats[0] - feats[1]
            feats_mean = feats.mean(dim=0)
            feats_var = feats.var(dim=0)
            feats_min, _ = feats.min(dim=0)
            feats_max, _ = feats.max(dim=0)

            '''feats_sum = feats.sum(dim=0)
            feats_sub = abs(feats[0] - feats[1])
            feats_prod = feats.prod(dim=0)
            feats_max, _ = feats.max(dim=0)'''
            
            #final_feat = torch.cat(
            #    [feats[0], feats[1], feats[0], feats[1]], dim=-1
            #    #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            #)

            final_feat = torch.cat(
                [euclidean_distance, feats_mean, feats_var, feats_min, feats_max], dim=-1
                #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            )

        out = self.dropout(final_feat)
        # out = self.fcfusion(out)
        # out = self.relu(out)
        out = self.fc(out)

        return out, feats[0], feats[1]
