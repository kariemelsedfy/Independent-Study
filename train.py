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
from model import Yolov1, YoloSegNet
from dataset import CubYoloDataset, CubSegDataset
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
    SEG_CHECKPOINT = "segnet_epoch200.pth.tar"  # whatever you saved

    # Recreate the model with same architecture
    CUB_CLASSES = 200
    NUM_SEG_CLASSES = 2


    test_dataset = CubSegDataset(
        cub_root=CUB_ROOT,
        seg_root=SEG_ROOT,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


    yolo = Yolov1(
        in_channels=3,
        split_size=7,
        num_boxes=2,
        num_classes=CUB_CLASSES,
    ).to(DEVICE)

    seg_model = YoloSegNet(yolo, num_seg_classes=NUM_SEG_CLASSES).to(DEVICE)

    # Load seg weights (no need for optimizer during testing)
    checkpoint = torch.load(SEG_CHECKPOINT, map_location=DEVICE)
    load_checkpoint(checkpoint, seg_model, optimizer=None)
    

    TEST_ONLY = True

    if TEST_ONLY:
        seg_model.eval()

        # get 1 batch from test loader
        images, masks = next(iter(test_loader))
        images = images.to(DEVICE)

        with torch.no_grad():
            logits = seg_model(images)
            preds = torch.argmax(logits, dim=1)

        import matplotlib.pyplot as plt

        # plot up to 4 images
        for i in range(min(10, images.size(0))):
            img   = images[i].permute(1, 2, 0).cpu().numpy()
            mask_gt   = masks[i].cpu().numpy()
            mask_pred = preds[i].cpu().numpy()

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
            plt.title("Pred mask")
            plt.imshow(mask_pred, cmap="gray")
            plt.axis("off")

            plt.show()

        return



if __name__ == "__main__":
    main()