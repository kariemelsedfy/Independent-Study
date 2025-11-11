import torch
import torch.nn as nn
import torch.nn.functional as F

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PixelHead(nn.Module):
    """
    Upsampling decoder that turns (B,1024,112,112) into (B,5,448,448)
    """
    def __init__(self, in_ch=1024, mid1=256, mid2=128, out_ch=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, mid1, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid1), nn.LeakyReLU(0.1, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid1, mid1, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid1), nn.LeakyReLU(0.1, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(mid1, mid2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid2), nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_conv = nn.Conv2d(mid2, out_ch, 1)

    def forward(self, x):
        # x: (B,1024,112,112)
        x = self.block1(x)                       # (B,256,112,112)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 224
        x = self.block2(x)                       # (B,256,224,224)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 448
        x = self.block3(x)                       # (B,128,448,448)
        x = self.out_conv(x)                     # (B,5,448,448)
        return x

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        # Replace FCs with pixel head returning 5×448×448
        self.pixel_head = PixelHead(in_ch=1024, out_ch=5)

    def forward(self, x):
        x = self.darknet(x)      # (B,1024,112,112) given your strides
        x = self.pixel_head(x)   # (B,5,448,448)
        return x

    def _create_conv_layers(self, architecture):
        layers, in_ch = [], self.in_channels
        for x in architecture:
            if isinstance(x, tuple):
                k, out_ch, s, p = x
                layers.append(CNNBlock(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
                in_ch = out_ch
            elif isinstance(x, str):
                # make pooling size-preserving
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            else:  # list with repeats
                conv1, conv2, reps = x
                for _ in range(reps):
                    layers.append(CNNBlock(in_ch, conv1[1],
                                           kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    layers.append(CNNBlock(conv1[1], conv2[1],
                                           kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_ch = conv2[1]
        return nn.Sequential(*layers)
