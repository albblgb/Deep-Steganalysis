""" This module creates SRNet model."""
import torch
from torch import Tensor
from torch import nn
# import sys
# sys.path.append('./srnet')
import config as c


class ConvBn(nn.Module):
    """Provides utility to create different types of layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor.
        Args:
            in_channels (int): no. of input channels.
            out_channels (int): no. of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns Conv2d followed by BatchNorm.

        Returns:
            Tensor: Output of Conv2D -> BN.
        """
        return self.batch_norm(self.conv(inp))


class Type1(nn.Module):
    """Creates type 1 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convbn = ConvBn(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 1 layer of SRNet.
        Args:
            inp (Tensor): input tensor.
        Returns:
            Tensor: Output of type 1 layer.
        """
        return self.relu(self.convbn(inp))


class Type2(nn.Module):
    """Creates type 2 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(in_channels, out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 2 layer of SRNet.
        Args:
            inp (Tensor): input tensor.
        Returns:
            Tensor: Output of type 2 layer.
        """
        return inp + self.convbn(self.type1(inp))


class Type3(nn.Module):
    """Creates type 3 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 3 layer of SRNet.
        Args:
            inp (Tensor): input tensor.

        Returns:
            Tensor: Output of type 3 layer.
        """
        out = self.batch_norm(self.conv1(inp))
        out1 = self.pool(self.convbn(self.type1(inp)))
        return out + out1


class Type4(nn.Module):
    """Creates type 4 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 4 layer of SRNet.
        Args:
            inp (Tensor): input tensor.
        Returns:
            Tensor: Output of type 4 layer.
        """
        return self.gap(self.convbn(self.type1(inp)))



class Model(nn.Module):
    """This is SRNet model class."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.type1s = nn.Sequential(Type1(c.stego_img_channel, 64), Type1(64, 16))
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
        )
        self.type3s = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.type4 = Type4(256, 512)
        self.dense = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns logits for input images.
        Args:
            inp (Tensor): input image tensor of shape (Batch, stego_img_channel, stego_img_height, stego_img_width)
        Returns:
            Tensor: Logits of shape (Batch, 2)
        """
 
        out = self.type1s(inp)
        out = self.type2s(out)
        out = self.type3s(out)
        out = self.type4(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out


# def weights_init(param) -> None:
#     """Initializes weights of Conv and fully connected."""

#     if isinstance(param, nn.Conv2d):
#         torch.nn.init.xavier_uniform_(param.weight.data)
#         if param.bias is not None:
#             torch.nn.init.constant_(param.bias.data, 0.2)
#     elif isinstance(param, nn.Linear):
#         torch.nn.init.normal_(param.weight.data, mean=0.0, std=0.01)
#         torch.nn.init.constant_(param.bias.data, 0.0)


if __name__ == "__main__":
    image = torch.randn((1, 1, 256, 256))
    net = Model()
    print(net(image).shape)
