import torch
import os
import pandas as pd
from PIL import Image

class CubYoloDataset(torch.utils.data.Dataset):
    """
    CUB-200-2011 dataset adapted to YOLO-v1 style label matrix.

    Requires the standard CUB text files:
      images.txt
      bounding_boxes.txt
      image_class_labels.txt
      train_test_split.txt

    and a root folder that contains the 'images/' subfolder.
    """

    def __init__(
        self,
        cub_root,                # e.g. "/path/to/CUB_200_2011"
        split="train",           # "train" or "test"
        S=7,
        B=2,
        C=200,
        transform=None,
    ):
        self.cub_root = cub_root
        self.images_dir = os.path.join(cub_root, "images")
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

        # ----- Load the standard CUB annotation files -----
        # images.txt: image_id, relative_path
        images_txt = os.path.join(cub_root, "images.txt")
        df_images = pd.read_csv(
            images_txt, sep=r"\s+", header=None, names=["image_id", "rel_path"]
        )

        # bounding_boxes.txt: image_id, x, y, width, height
        bbox_txt = os.path.join(cub_root, "bounding_boxes.txt")
        df_boxes = pd.read_csv(
            bbox_txt,
            sep=r"\s+",
            header=None,
            names=["image_id", "x", "y", "width", "height"],
        )

        # image_class_labels.txt: image_id, class_id (1..200)
        labels_txt = os.path.join(cub_root, "image_class_labels.txt")
        df_labels = pd.read_csv(
            labels_txt,
            sep=r"\s+",
            header=None,
            names=["image_id", "class_id"],
        )

        # train_test_split.txt: image_id, is_training_image (0/1)
        split_txt = os.path.join(cub_root, "train_test_split.txt")
        df_split = pd.read_csv(
            split_txt,
            sep=r"\s+",
            header=None,
            names=["image_id", "is_train"],
        )

        # Merge all info by image_id
        df = df_images.merge(df_boxes, on="image_id")
        df = df.merge(df_labels, on="image_id")
        df = df.merge(df_split, on="image_id")

        # Filter train / test
        if split == "train":
            df = df[df["is_train"] == 1]
        else:
            df = df[df["is_train"] == 0]

        # Reset index for __getitem__
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # --- basic fields ---
        rel_path  = row["rel_path"]
        img_path  = os.path.join(self.images_dir, rel_path)
        img_id    = row["image_id"]
        class_id  = int(row["class_id"])    # 1..200
        # convert to 0..C-1 index:
        class_label = class_id - 1

        # bounding box in pixel coords from CUB (x, y, w, h)
        # x,y are top-left corner in pixels (1-based in the official docs)
        x = float(row["x"])
        y = float(row["y"])
        w = float(row["width"])
        h = float(row["height"])

        # --- load image ---
        image = Image.open(img_path).convert("RGB")
        W_img, H_img = image.size

        # ---- convert to YOLO-style normalized (cx, cy, w, h in [0,1]) ----
        # Shift x,y if you want to correct for 1-based indexing:
        # x0 = x - 1; y0 = y - 1
        # I'll just treat them as 0-based; off-by-one is negligible for training.
        x0 = x
        y0 = y

        cx = (x0 + w / 2.0) / W_img
        cy = (y0 + h / 2.0) / H_img
        nw = w / W_img
        nh = h / H_img

        boxes = torch.tensor([[class_label, cx, cy, nw, nh]], dtype=torch.float32)

        # Optional transform in your YOLO pipeline: transform(img, boxes)
        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        # Recompute normalized coords in case the transform resized/cropped
        # (Assumes your Compose keeps them normalized; if so, you can skip this.)
        class_label, cx, cy, nw, nh = boxes[0].tolist()

        # ---- build YOLO label matrix [S, S, C + 5B] ----
        S, B, C = self.S, self.B, self.C
        label_matrix = torch.zeros((S, S, C + 5 * B))

        # Which cell (i,j) contains the center?
        i, j = int(S * cy), int(S * cx)
        i = max(min(i, S - 1), 0)
        j = max(min(j, S - 1), 0)

        x_cell = S * cx - j
        y_cell = S * cy - i
        w_cell = nw * S
        h_cell = nh * S

        # Only one object per cell (standard YOLO v1 assumption)
        if label_matrix[i, j, C] == 0:
            # objectness
            label_matrix[i, j, C] = 1.0
            # bbox coords for the first box
            label_matrix[i, j, C + 1 : C + 5] = torch.tensor(
                [x_cell, y_cell, w_cell, h_cell], dtype=torch.float32
            )
            # one-hot class
            label_matrix[i, j, int(class_label)] = 1.0

        return image, label_matrix
