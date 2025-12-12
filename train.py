"""
Main file for training Yolo model on CUB dataset

"""
import matplotlib.pyplot as plt
import sys
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Yolov1, YoloUNet, YoloSegNet, Yolov1seq
from dataset import CubYoloDataset, CubSegDataset
import os
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    batch_iou_from_logits
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
EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = False
LOAD_MODEL = True
LOAD_MODEL_FILE = "CUB2.pth.tar"
CUB_ROOT = "CUB_200_2011/CUB_200_2011"
SEG_ROOT = "CUB_200_2011/segmentations"  # change if path is different


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


def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader)
    losses = []

    for x, y in loop:
        x = x.to(DEVICE)
        y = y.long().to(DEVICE)   # masks

        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return sum(losses) / len(losses)

def eval_fn(loader, model, loss_fn):
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.long().to(DEVICE)

            preds = model(x)
            loss = loss_fn(preds, y)
            losses.append(loss.item())

    return sum(losses) / len(losses)

def main():
    SEG_CHECKPOINT = "segnet_epoch200.pth.tar"  # whatever you saved

    # # Recreate the model with same architecture
    CUB_CLASSES = 200
    NUM_SEG_CLASSES = 2

    train_dataset = CubSegDataset(
        cub_root=CUB_ROOT,
        seg_root=SEG_ROOT,
        split="train",
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_dataset = CubSegDataset(
        cub_root=CUB_ROOT,
        seg_root=SEG_ROOT,
        split="test",
        transform = transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


    # 1) Create YOLO encoder
    yolo = Yolov1seq(
        in_channels=3,
        split_size=7,
        num_boxes=2,
        num_classes=CUB_CLASSES,
    ).to(DEVICE)

    # 2) Load YOLO encoder weights ONLY
    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_MODEL_FILE, map_location=DEVICE)
        load_checkpoint(checkpoint, yolo, optimizer=None)
        print("✅ Loaded pretrained YOLO encoder weights")

    # 3) Create the segmentation model
    seg_model = YoloSegNet(yolo, num_seg_classes=NUM_SEG_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        seg_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # seg_model = YoloSegNet(yolo, num_seg_classes=NUM_SEG_CLASSES).to(DEVICE)

    # # Load seg weights (no need for optimizer during testing)
    # checkpoint = torch.load(SEG_CHECKPOINT, map_location=DEVICE)
    # load_checkpoint(checkpoint, seg_model, optimizer=None)
    

    TEST_ONLY = True

    if TEST_ONLY:

        checkpoint = torch.load(SEG_CHECKPOINT, map_location=DEVICE)
        load_checkpoint(checkpoint, seg_model, optimizer=None)
        # print("✅ Loaded trained UNet segmentation model")
        seg_model.eval()
        save_dir = "decoderNoSkipPlots"   
        os.makedirs(save_dir, exist_ok=True)

        # get 1 batch from test loader
        images, masks = next(iter(test_loader))
        images = images.to(DEVICE)

        with torch.no_grad():
            logits = seg_model(images)
            preds = torch.argmax(logits, dim=1)

        # plot up to 4 images
        for i in range(min(10, images.size(0))):
            img   = images[i].permute(1, 2, 0).cpu().numpy()
            mask_gt   = masks[i].cpu().numpy()
            mask_pred = preds[i].cpu().numpy()

            # ---- IoU for this sample (foreground class=1) ----
            pred_bool = (mask_pred == 1)
            gt_bool   = (mask_gt == 1)

            intersection = (pred_bool & gt_bool).sum()
            union        = (pred_bool | gt_bool).sum()
            iou = (intersection / union) if union > 0 else 0.0

            plt.figure(figsize=(12,4))

            plt.subplot(1,3,1)
            plt.title("Image")
            plt.imshow(img)
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.title("GT mask")
            plt.imshow(mask_gt, cmap="gray")
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.title(f"Pred mask\nIoU = {iou:.3f}")
            plt.imshow(mask_pred, cmap="gray")
            plt.axis("off")

            out_path = os.path.join(save_dir, f"decoderSkip_{i}.png")
            plt.savefig(out_path, bbox_inches="tight")


            plt.show()

        return


    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss = train_fn(
            train_loader,
            seg_model,
            optimizer,
            criterion,
        )

        val_loss = eval_fn(
            test_loader,
            seg_model,
            criterion,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        if epoch == 49:
            save_checkpoint(
                {
                    "state_dict": seg_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                filename="UnetSegmentation.pth.tar",
            )


if __name__ == "__main__":
    main()