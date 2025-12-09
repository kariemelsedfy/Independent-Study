"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
Now uses a conv head (no FCs) and supports dense stride-1 inference.
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

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
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=1):
        """
        split_size: S (e.g. 7)
        num_boxes:  B (e.g. 2)
        num_classes: C
        """
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.darknet = self._create_conv_layers(self.architecture)

        # Conv head instead of fully-connected layers:
        # per cell we predict C class scores + B*(x,y,w,h,conf)
        out_channels = self.C + self.B * 5
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # Pass through each layer manually since darknet is a ModuleList
        for layer in self.darknet:
            x = layer(x)

        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.size(0), -1)

    def forward_features(self, x):
        skip_connections = []

        for layer in self.darknet:
            x = layer(x)

            # save skips right after each MaxPool
            if isinstance(layer, nn.MaxPool2d):
                skip_connections.append(x)

        return x, skip_connections

    # def forward_dense_stride1(self, x):
    #     """
    #     Dense inference mode:
    #     - Temporarily set all Conv2d & MaxPool2d strides to 1
    #     - Replace pooling kernels with 1x1 (no spatial downsample)
    #     - Run the network and restore original settings

    #     Returns: (B, C+5B, H, W) where H,W are ~input size
    #     """
    #     # Save original strides / kernel sizes / paddings
    #     conv_modules = []
    #     pool_modules = []

    #     for m in self.darknet.modules():
    #         if isinstance(m, nn.Conv2d):
    #             conv_modules.append((m, m.stride))
    #         elif isinstance(m, nn.MaxPool2d):
    #             pool_modules.append((m, m.kernel_size, m.stride, m.padding))

    #     # Patch: remove downsampling
    #     for m, stride in conv_modules:
    #         if stride != (1, 1):
    #             m.stride = (1, 1)

    #     for m, k, s, p in pool_modules:
    #         # Make pooling a 1x1 identity-like op
    #         m.kernel_size = (1, 1)
    #         m.stride = (1, 1)
    #         m.padding = (0, 0)

    #     # Forward pass with no grad
    #     with torch.no_grad():
    #         feat = self.darknet(x)   # (B,1024,H,W) now close to input size
    #         out = self.head(feat)    # (B,C+5B,H,W)

    #     # Restore original conv/pool settings
    #     for (m, stride) in conv_modules:
    #         m.stride = stride

    #     for (m, k, s, p) in pool_modules:
    #         m.kernel_size = k
    #         m.stride = s
    #         m.padding = p

    #     return out  # (B, C+5B, H, W)

    def _create_conv_layers(self, architecture):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels, x[1],
                        kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                )
                in_channels = x[1]

            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif type(x) == list:
                conv1, conv2, num_repeats = x
                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    )
                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    )
                    in_channels = conv2[1]

        return layers



class YoloUNet(nn.Module):
    def __init__(self, yolo_encoder: Yolov1, num_seg_classes: int):
        super().__init__()

        self.encoder = yolo_encoder

        # Freeze YOLO encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # YOLO bottleneck has 1024 channels at 7x7
        # Decoder: 7 -> 14 -> 28 -> 56 -> 112 -> 224 -> 448
        # Skips at: 14 (s4), 28 (s3), 56 (s2), 112 (s1)

        self.up1 = self._up(1024, 512)           # 7  -> 14, no skip yet
        self.up2 = self._up(512 + 1024, 512)     # 14 -> 28, concat s4 (1024) → 1536 in
        self.up3 = self._up(512 + 512, 256)      # 28 -> 56, concat s3 (512)  → 1024 in
        self.up4 = self._up(256 + 192, 128)      # 56 -> 112, concat s2 (192) → 448 in
        self.up5 = self._up(128 + 64, 64)        # 112 -> 224, concat s1 (64) → 192 in

        self.out_up = self._up(64, 32)           # 224 -> 448, no skip
        self.seg_head = nn.Conv2d(32, num_seg_classes, kernel_size=1)

    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # bottleneck: (B,1024,7,7)
        # skips: [s1(64,112,112), s2(192,56,56), s3(512,28,28), s4(1024,14,14)]
        bottleneck, skips = self.encoder.forward_features(x)
        s1, s2, s3, s4 = skips

        # 7 -> 14
        x = self.up1(bottleneck)         # (B,512,14,14)

        # 14 -> 28 + skip 14
        x = torch.cat([x, s4], dim=1)    # (B,512+1024=1536,14,14)
        x = self.up2(x)                  # (B,512,28,28)

        # 28 -> 56 + skip 28
        x = torch.cat([x, s3], dim=1)    # (B,512+512=1024,28,28)
        x = self.up3(x)                  # (B,256,56,56)

        # 56 -> 112 + skip 56
        x = torch.cat([x, s2], dim=1)    # (B,256+192=448,56,56)
        x = self.up4(x)                  # (B,128,112,112)

        # 112 -> 224 + skip 112
        x = torch.cat([x, s1], dim=1)    # (B,128+64=192,112,112)
        x = self.up5(x)                  # (B,64,224,224)

        # 224 -> 448 (no skip)
        x = self.out_up(x)               # (B,32,448,448)

        return self.seg_head(x)          # (B,num_seg_classes,448,448)