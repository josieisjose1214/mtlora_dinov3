import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        t = targets.clamp(min=0)
        one_hot = F.one_hot(t, num_classes + 1)[:, 1:].float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
        p_t = p * one_hot + (1 - p) * (1 - one_hot)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * one_hot + (1 - self.alpha) * (1 - one_hot)
        return (alpha_t * focal_weight * ce).sum()


class IOULoss(nn.Module):
    def __init__(self, loss_type: str = "iou"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
        target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
        w_inter = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
        h_inter = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])
        area_inter = w_inter * h_inter
        area_union = pred_area + target_area - area_inter
        ious = (area_inter + 1.0) / (area_union + 1.0)
        if self.loss_type == "linear_iou":
            losses = 1 - ious
        else:
            losses = -torch.log(ious)
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        return losses.sum()


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 1, num_convs: int = 4, num_levels: int = 4, prior_prob: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        cls_tower = []
        bbox_tower = []
        for _ in range(num_convs):
            cls_tower.extend([nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True), nn.GroupNorm(32, in_channels), nn.ReLU()])
            bbox_tower.extend([nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True), nn.GroupNorm(32, in_channels), nn.ReLU()])
        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        for modules in [self.cls_tower, self.bbox_tower]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        for layer in [self.cls_logits, self.bbox_pred, self.centerness]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features: List[torch.Tensor]):
        logits, bbox_reg, centerness = [], [], []
        for level, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            bbox_feat = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_feat))
            centerness.append(self.centerness(cls_feat))
            bbox_reg.append(F.relu(self.scales[level](self.bbox_pred(bbox_feat))))
        return logits, bbox_reg, centerness


def compute_locations(features: List[torch.Tensor], fpn_strides: List[int]):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.shape[-2:]
        stride = fpn_strides[level]
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=feature.device)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=feature.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        locations.append(torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1) + stride // 2)
    return locations


class FCOSLoss(nn.Module):
    def __init__(self, num_classes: int = 1, num_levels: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.focal_loss = SigmoidFocalLoss(gamma=2.0, alpha=0.25)
        self.iou_loss = IOULoss(loss_type="iou")
        self.centerness_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    @staticmethod
    def compute_centerness(reg_targets: torch.Tensor):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0].clamp(min=1e-6)) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0].clamp(min=1e-6)
        )
        return torch.sqrt(centerness.clamp(min=0))

    @staticmethod
    def _compute_targets_per_level(locations: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, obj_size: List[float]):
        m = locations.shape[0]
        g = boxes.shape[0]
        device = locations.device
        if g == 0:
            return torch.zeros(m, dtype=torch.long, device=device), torch.zeros(m, 4, dtype=torch.float32, device=device)

        xs, ys = locations[:, 0], locations[:, 1]
        l = xs[:, None] - boxes[:, 0][None]
        t = ys[:, None] - boxes[:, 1][None]
        r = boxes[:, 2][None] - xs[:, None]
        b = boxes[:, 3][None] - ys[:, None]
        reg_targets = torch.stack([l, t, r, b], dim=2)

        is_in_boxes = reg_targets.min(dim=2)[0] > 0
        max_reg = reg_targets.max(dim=2)[0]
        is_cared = (max_reg >= obj_size[0]) & (max_reg <= obj_size[1])
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        loc_to_gt_area = box_areas[None].expand(m, g).clone()
        loc_to_gt_area[~is_in_boxes] = 1e8
        loc_to_gt_area[~is_cared] = 1e8
        min_area, gt_inds = loc_to_gt_area.min(dim=1)
        reg_targets_per_loc = reg_targets[torch.arange(m, device=device), gt_inds]
        labels_per_loc = labels[gt_inds].clone()
        labels_per_loc[min_area >= 1e8] = 0
        return labels_per_loc, reg_targets_per_loc

    def _to_fcos_targets(self, targets, image_size, device):
        if isinstance(targets, dict) and {"batch_idx", "cls", "bboxes"}.issubset(targets.keys()):
            h, w = image_size
            batch_idx = targets["batch_idx"].to(device=device)
            cls = targets["cls"].to(device=device).long()
            bboxes = targets["bboxes"].to(device=device).float()
            if bboxes.numel() > 0 and bboxes.max() <= 1.5:
                bboxes = bboxes.clone()
                bboxes[:, 0] *= w
                bboxes[:, 1] *= h
                bboxes[:, 2] *= w
                bboxes[:, 3] *= h
            x1 = bboxes[:, 0] - bboxes[:, 2] / 2
            y1 = bboxes[:, 1] - bboxes[:, 3] / 2
            x2 = bboxes[:, 0] + bboxes[:, 2] / 2
            y2 = bboxes[:, 1] + bboxes[:, 3] / 2
            xyxy = torch.stack([x1, y1, x2, y2], dim=1) if bboxes.numel() > 0 else bboxes.new_zeros((0, 4))

            if batch_idx.numel() == 0:
                return [{"boxes": xyxy.new_zeros((0, 4)), "labels": cls.new_zeros((0,), dtype=torch.long)}]
            batch_size = int(batch_idx.max().item()) + 1
            out = []
            for i in range(batch_size):
                mask = batch_idx == i
                out.append({"boxes": xyxy[mask], "labels": cls[mask] + 1})
            return out
        return targets

    def forward(self, output: Dict, targets):
        logits = output["logits"]
        bbox_reg = output["bbox_reg"]
        centerness = output["centerness"]
        locations = output["locations"]
        image_size = output.get("image_size", (448, 448))
        targets = self._to_fcos_targets(targets, image_size=image_size, device=logits[0].device)

        b = logits[0].shape[0]
        object_sizes = [[-1, 64], [64, 128], [128, 256], [256, 1e8]][: self.num_levels]
        labels_level_first = []
        reg_targets_level_first = []
        for level_idx in range(self.num_levels):
            labels_per_level = []
            reg_per_level = []
            for img_idx in range(b):
                boxes = targets[img_idx]["boxes"]
                gt_labels = targets[img_idx]["labels"]
                lab, reg = self._compute_targets_per_level(locations[level_idx], boxes, gt_labels, object_sizes[level_idx])
                labels_per_level.append(lab)
                reg_per_level.append(reg)
            labels_level_first.append(torch.cat(labels_per_level))
            reg_targets_level_first.append(torch.cat(reg_per_level))

        logits_flat = torch.cat([l.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for l in logits])
        bbox_reg_flat = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in bbox_reg])
        centerness_flat = torch.cat([c.reshape(-1) for c in centerness])
        labels_flat = torch.cat(labels_level_first)
        reg_targets_flat = torch.cat(reg_targets_level_first)
        pos_inds = labels_flat > 0
        num_pos = pos_inds.sum().clamp(min=1).float()
        cls_loss = self.focal_loss(logits_flat, labels_flat.long()) / num_pos
        if pos_inds.sum() > 0:
            ct = self.compute_centerness(reg_targets_flat[pos_inds])
            reg_loss = self.iou_loss(bbox_reg_flat[pos_inds], reg_targets_flat[pos_inds], ct) / ct.sum().clamp(min=1)
            centerness_loss = self.centerness_loss_fn(centerness_flat[pos_inds], ct) / num_pos
        else:
            reg_loss = bbox_reg_flat.sum() * 0
            centerness_loss = centerness_flat.sum() * 0
        return cls_loss + reg_loss + centerness_loss


@torch.no_grad()
def fcos_inference(output: Dict, score_thresh: float = 0.5, nms_thresh: float = 0.6, max_detections: int = 100):
    logits = output["logits"]
    bbox_reg = output["bbox_reg"]
    centerness = output["centerness"]
    locations = output["locations"]
    batch_size = logits[0].shape[0]
    results = []
    for img_idx in range(batch_size):
        boxes_all, scores_all, labels_all = [], [], []
        for logit, bbox, center, loc in zip(logits, bbox_reg, centerness, locations):
            c = logit.shape[1]
            scores = torch.sigmoid(logit[img_idx]).permute(1, 2, 0).reshape(-1, c)
            center_scores = torch.sigmoid(center[img_idx]).reshape(-1)
            bbox_pred = bbox[img_idx].permute(1, 2, 0).reshape(-1, 4)
            scores = scores * center_scores[:, None]
            max_scores, max_cls = scores.max(dim=1)
            keep = max_scores > score_thresh
            if keep.sum() == 0:
                continue
            s = max_scores[keep]
            cls = max_cls[keep] + 1
            bp = bbox_pred[keep]
            lk = loc[keep]
            boxes = torch.stack(
                [(lk[:, 0] - bp[:, 0]).clamp(min=0), (lk[:, 1] - bp[:, 1]).clamp(min=0), lk[:, 0] + bp[:, 2], lk[:, 1] + bp[:, 3]],
                dim=1,
            )
            boxes_all.append(boxes)
            scores_all.append(s)
            labels_all.append(cls.float() - 1.0)
        if len(boxes_all) == 0:
            results.append(torch.empty((0, 6), device=logits[0].device))
            continue
        boxes_all = torch.cat(boxes_all)
        scores_all = torch.cat(scores_all)
        labels_all = torch.cat(labels_all)
        keep = nms(boxes_all, scores_all, nms_thresh)[:max_detections]
        det = torch.cat([boxes_all[keep], scores_all[keep, None], labels_all[keep, None]], dim=1)
        results.append(det)
    return results
