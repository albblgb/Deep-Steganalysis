"""
This is unofficial implementation of XuNet: Structural Design of Convolutional
Neural Networks for Steganalysis . """
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import config as c


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor."""

    def __init__(self) -> None:
        """Constructor"""

        super().__init__()
        # pylint: disable=E1101
        self.kv_filter = nn.Parameter(
            torch.tensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ],
            ).view(1, 1, 5, 5)
            / 12.0
        )  # pylint: enable=E1101

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""

        for i in range(c.stego_img_channel):
            if i == 0:
                features = F.conv2d(inp[:, i, :, :].unsqueeze(dim=1), self.kv_filter, stride=1, padding=2)
            else:
                features = torch.cat((features, F.conv2d(inp[:, i, :, :].unsqueeze(dim=1), self.kv_filter, stride=1, padding=2)), dim=1)
            
        return features
        # return F.conv2d(inp, self.kv_filter, stride=1, padding=2)


class ConvBlock(nn.Module):
    """This class returns building block for XuNet class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        abs: str = False,
    ) -> None:
        super().__init__()

        if kernel_size == 5:
            self.padding = 2
        else:
            self.padding = 0

        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.abs = abs
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv->batch_norm."""
        if self.abs:
            return self.pool(
                self.activation(self.batch_norm(torch.abs(self.conv(inp))))
            )
        return self.pool(self.activation(self.batch_norm(self.conv(inp))))


class XuNet(nn.Module):
    """This class returns XuNet model."""

    def __init__(self) -> None:
        super().__init__()

        self.ImageProcessingLayer = ImageProcessing()
        self.layer1 = ConvBlock(
            c.stego_img_channel, 8, kernel_size=5, activation="tanh", abs=True
        )
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=2),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Returns logit for the given tensor."""
        with torch.no_grad():
            out = self.ImageProcessingLayer(image)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        return out


if __name__ == "__main__":
    net = XuNet()
    print(net)
    inp_image = torch.randn((1, 1, 512, 512))
    print(net(inp_image))