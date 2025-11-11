"""
YOLOv1 Loss for single-class datasets (C=1).
Per-cell layout (B=2):
  [ class(0) | conf1(1) | x1(2) y1(3) w1(4) h1(5) | conf2(6) | x2(7) y2(8) w2(9) h2(10) ]
"""

import torch
import torch.nn as nn
from utils import intersection_over_union

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLoss(nn.Module):
    """
    pred:   (B,5,H,W) -> [logit_mask, x1, y1, x2, y2]
    target: (B,5,H,W) -> [mask in {0,1}, xyxy normalized 0..1]
    """
    def __init__(self, lambda_box: float = 5.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        mask = target[:, 0:1]        # (B,1,H,W)
        gt   = target[:, 1:5]        # (B,4,H,W)

        # blob (inside-box) supervision
        loss_blob = self.bce(pred[:, 0:1], mask)

        # coords (only where mask==1)
        pred_xyxy = torch.sigmoid(pred[:, 1:5])   # map to [0,1]
        l1 = F.smooth_l1_loss(pred_xyxy, gt, reduction="none").mean(dim=1, keepdim=True)  # (B,1,H,W)
        loss_box = (l1 * mask).sum() / (mask.sum() + 1e-6)

        return loss_blob + self.lambda_box * loss_box


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C  # C=1
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, target):
        """
        predictions: (N, S*S*(C + B*5)) -> (N, S, S, C + B*5) == (N,S,S,11) when C=1,B=2
        target:      (N, S, S, C + B*5)
        """
        # reshape to grid
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # indices (C=1)
        class_slice = slice(0, 1)   # [0]
        conf1_slice = slice(1, 2)   # [1:2]
        box1_slice  = slice(2, 6)   # [2:6]
        conf2_slice = slice(6, 7)   # [6:7]
        box2_slice  = slice(7, 11)  # [7:11]

        # ---- robust IoU + best-box selection (avoids dim mixups) ----
        N = predictions.shape[0]
        pb1 = predictions[..., box1_slice].reshape(-1, 4)   # (N*S*S,4)
        pb2 = predictions[..., box2_slice].reshape(-1, 4)   # (N*S*S,4)
        tb  = target[...,      box1_slice].reshape(-1, 4)   # (N*S*S,4)

        iou_b1 = intersection_over_union(pb1, tb).reshape(N, self.S, self.S)  # (N,S,S)
        iou_b2 = intersection_over_union(pb2, tb).reshape(N, self.S, self.S)  # (N,S,S)

        ious      = torch.stack((iou_b1, iou_b2), dim=-1)   # (N,S,S,2)
        bestbox   = ious.argmax(dim=-1)                     # (N,S,S) in {0,1}
        bestbox_u = bestbox.unsqueeze(-1).float()           # (N,S,S,1)

        # objectness mask (same slot as conf1)
        exists_box = target[..., conf1_slice]  # (N,S,S,1)

        # 1) Coordinate loss (only where object exists)
        box_predictions = exists_box * (
            bestbox_u * predictions[..., box2_slice] +
            (1.0 - bestbox_u) * predictions[..., box1_slice]
        )
        box_targets = exists_box * target[..., box1_slice]

        # sqrt on w,h for stability
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(torch.clamp(box_targets[..., 2:4], min=1e-6))

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets,     end_dim=-2),
        )

        # 2) Objectness loss (confidence of best box)
        pred_conf_best = bestbox_u * predictions[..., conf2_slice] + (1.0 - bestbox_u) * predictions[..., conf1_slice]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_conf_best),
            torch.flatten(exists_box * target[..., conf1_slice]),
        )

        # 3) No-object loss (both boxes where no object)
        noobj = 1.0 - exists_box
        no_object_loss = self.mse(
            torch.flatten(noobj * predictions[..., conf1_slice], start_dim=1),
            torch.flatten(noobj * target[...,     conf1_slice], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten(noobj * predictions[..., conf2_slice], start_dim=1),
            torch.flatten(noobj * target[...,     conf1_slice], start_dim=1),
        )

        # 4) Class loss (only where object exists) — with C=1 this is a single logit
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., class_slice], end_dim=-2),
            torch.flatten(exists_box * target[...,     class_slice], end_dim=-2),
        )

        # total
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss
