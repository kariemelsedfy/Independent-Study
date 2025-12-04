"""
Main file for training Yolo model on bus dataset

"""
import matplotlib.pyplot as plt
import sys
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import CubYoloDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 30
NUM_WORKERS = 0
PIN_MEMORY = False
LOAD_MODEL = True
LOAD_MODEL_FILE = "CUB2.pth.tar"
CUB_ROOT = "CUB_200_2011/CUB_200_2011"


from torchvision import transforms as T
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        # strongly encourage ToTensor last
        if any(isinstance(t, T.ToTensor) for t in transforms):
            assert isinstance(self.transforms[-1], T.ToTensor), \
                "Place ToTensor() as the LAST transform."

    def __call__(self, img, bboxes):
        for t in self.transforms:
            # If img is already a tensor, skip PIL-only transforms and ToTensor
            if isinstance(img, torch.Tensor):
                if isinstance(t, T.ToTensor):
                    continue
                # Skip transforms that expect PIL images
                if isinstance(t, (T.Resize, T.CenterCrop, T.RandomHorizontalFlip,
                                  T.ColorJitter, T.RandomResizedCrop)):
                    continue
            img = t(img)
        return img, bboxes



transform = transform = Compose([
    T.Resize((448, 448)),
    # add more PIL transforms here if you want (ColorJitter, RandomHorizontalFlip, etc.)
    T.ToTensor(),   # <- LAST and only once
])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    S = 7
    B = 2
    C = 200

    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # --- DATASETS & LOADERS (same as during training) ---
    train_dataset = CubYoloDataset(
        cub_root=CUB_ROOT,
        split="train",
        S=S,
        B=B,
        C=C,
        transform=transform,
    )

    test_dataset = CubYoloDataset(
        cub_root=CUB_ROOT,
        split="test",
        S=S,
        B=B,
        C=C,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    TEST_ONLY = True

    if TEST_ONLY:
        # 1) Load your trained weights
        checkpoint = torch.load(LOAD_MODEL_FILE, map_location=DEVICE)
        load_checkpoint(checkpoint, model, optimizer=None)  # ignore optimizer

        model.eval()
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_loader):
                x = x.to(DEVICE)
                labels = labels.to(DEVICE)

                # 2) Forward pass
                predictions = model(x)  # shape: (B, S*S*(C+5B))

                # 3) Convert to bounding boxes per image
                #    each element in bboxes[idx] is [class, conf, x, y, w, h]
                bboxes = cellboxes_to_boxes(predictions)

                # (optional) also decode ground-truth boxes if you want to see them
                # true_bboxes = cellboxes_to_boxes(labels)

                # 4) For a few images in the batch, apply NMS and plot
                batch_size = x.shape[0]
                for idx in range(min(batch_size, 10)):   # show up to 4 images
                    nms_boxes = non_max_suppression(
                        bboxes[idx],
                        iou_threshold=0.5,
                        threshold=0.4,
                        box_format="midpoint",  # because boxes are [x, y, w, h]
                    )

                    # x[idx] is (3,H,W) → convert to (H,W,3) and move to CPU
                    img = x[idx].permute(1, 2, 0).to("cpu")

                    # 5) Draw predicted boxes on the image
                    plot_image(img, nms_boxes)
                    # if you also want GT, you could call plot_image twice or modify it

                # only first batch
                break

        return
if __name__ == "__main__":
    main()