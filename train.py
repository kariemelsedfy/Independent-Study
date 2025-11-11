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
from dataset import busDataset
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
EPOCHS = 20
NUM_WORKERS = 0
PIN_MEMORY = False
LOAD_MODEL = False
LOAD_MODEL_FILE = "checkpoint.pth.tar"
IMG_DIR = "bus_dataset/images"
LABEL_DIR = "bus_dataset/labels"


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
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location='cpu'), model, optimizer)
    

    train_dataset = busDataset(
        "splits/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = busDataset(
        "splits/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
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

        # model.eval()
        # with torch.no_grad():
        #     for imgs, _ in test_loader:
        #         imgs = imgs.to(DEVICE)
        #         preds = model(imgs)
        #         probs = preds[:,0].sigmoid().cpu()   # (B,448,448)

        #         img0 = FT.to_pil_image(imgs[0].cpu())
        #         m0   = probs[0].numpy()

        #         plt.figure(figsize=(6,6))
        #         plt.imshow(img0)
        #         plt.imshow(m0, alpha=0.5, cmap="inferno")  # heatmap overlay

        #         # plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
        #         #                                 fill=False, edgecolor="lime", linewidth=2))
        #         plt.axis("off")
        #         plt.show()
        # return

    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #         plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #     import sys
        #     sys.exit()

        model.train()
        loop = tqdm(train_loader, leave=True)
        running = 0.0

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)   # y is (B,5,H,W)
            out = model(x)                      # (B,5,H,W)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}: loss={running/len(loop):.4f}")

        if epoch == 49:
            save_checkpoint()

if __name__ == "__main__":
    main()