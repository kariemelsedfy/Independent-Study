"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image

from PIL import Image, ImageOps

class busDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=1, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # ---- read labels ----
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        boxes = torch.tensor(boxes)

        # ---- read image & force RGB ----
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)         # fix orientation from EXIF
        img = img.convert("RGB")                   # <<< guarantees 3 channels

        # ---- transforms ----
        if self.transform:
            img, boxes = self.transform(img, boxes)

        # ---- grid encoding ----
        # Remove the 7x7 label matrix construction block entirely and replace with:
        H = W = 448
        target = torch.zeros(5, H, W)  # [mask, x1, y1, x2, y2]

        # boxes is Nx5 with columns: class, cx, cy, w, h (normalized 0..1)
        for box in boxes:
            class_label, cx, cy, w, h = box.tolist()
            x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            target[0, y1:y2, x1:x2] = 1.0
            target[1, y1:y2, x1:x2] = x1 / W
            target[2, y1:y2, x1:x2] = y1 / H
            target[3, y1:y2, x1:x2] = x2 / W
            target[4, y1:y2, x1:x2] = y2 / H

        return img, target

