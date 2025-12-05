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
        """
        Standard YOLO forward used for training.
        Outputs shape: (B, S*S*(C + B*5))
        which matches YoloLoss and your dataset label_matrix.
        """
        x = self.darknet(x)             # (B,1024,S,S), e.g. (B,1024,7,7) for 448x448 input
        x = self.head(x)                # (B,C+5B,S,S)
        x = x.permute(0, 2, 3, 1)       # (B,S,S,C+5B)
        return x.reshape(x.size(0), -1) # (B, S*S*(C+5B))

    def forward_dense_stride1(self, x):
        """
        Dense inference mode:
        - Temporarily set all Conv2d & MaxPool2d strides to 1
        - Replace pooling kernels with 1x1 (no spatial downsample)
        - Run the network and restore original settings

        Returns: (B, C+5B, H, W) where H,W are ~input size
        """
        # Save original strides / kernel sizes / paddings
        conv_modules = []
        pool_modules = []

        for m in self.darknet.modules():
            if isinstance(m, nn.Conv2d):
                conv_modules.append((m, m.stride))
            elif isinstance(m, nn.MaxPool2d):
                pool_modules.append((m, m.kernel_size, m.stride, m.padding))

        # Patch: remove downsampling
        for m, stride in conv_modules:
            if stride != (1, 1):
                m.stride = (1, 1)

        for m, k, s, p in pool_modules:
            # Make pooling a 1x1 identity-like op
            m.kernel_size = (1, 1)
            m.stride = (1, 1)
            m.padding = (0, 0)

        # Forward pass with no grad
        with torch.no_grad():
            feat = self.darknet(x)   # (B,1024,H,W) now close to input size
            out = self.head(feat)    # (B,C+5B,H,W)

        # Restore original conv/pool settings
        for (m, stride) in conv_modules:
            m.stride = stride

        for (m, k, s, p) in pool_modules:
            m.kernel_size = k
            m.stride = s
            m.padding = p

        return out  # (B, C+5B, H, W)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)



class YoloSegNet(nn.Module):
    def __init__(self, yolo_encoder: Yolov1, num_seg_classes: int):
        super().__init__()

        # 1) Use YOLO's darknet as encoder
        self.encoder = yolo_encoder.darknet  # (B, 1024, 7, 7) for 448×448 input

        # 2) Freeze encoder weights
        for p in self.encoder.parameters():
            p.requires_grad = False

        enc_channels = 1024  # from your architecture

        # 3) SegNet-like decoder: 7×7 → 448×448 (factor 64 = 2^6)
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(enc_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 224x224 -> 448x448, output logits for each class
            nn.ConvTranspose2d(32, num_seg_classes, kernel_size=2, stride=2),
            # no activation here – use logits with CrossEntropyLoss
        )

    def forward(self, x):
        # Encoder: frozen YOLO backbone
        feat = self.encoder(x)           # (B, 1024, 7, 7)

        # Decoder: trainable segmentation head
        logits = self.decoder(feat)      # (B, num_seg_classes, 448, 448)

        return logits