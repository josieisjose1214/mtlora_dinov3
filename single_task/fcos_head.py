import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import nms


class SigmoidFocalLoss(nn.Module):
    """Pure PyTorch Sigmoid Focal Loss (matches FCOS official implementation)"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        logits: (N, num_classes) raw logits
        targets: (N,) class indices, 0=background, 1..C=foreground classes
        """
        num_classes = logits.shape[1]

        # One-hot: class 0 (bg) -> all zeros; class k (fg) -> column k-1 = 1
        t = targets.clamp(min=0)
        one_hot = F.one_hot(t, num_classes + 1)[:, 1:].float()  # (N, C)

        p = torch.sigmoid(logits)

        # Per-element binary cross entropy with correct one-hot target
        ce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction='none')

        # p_t: probability assigned to the correct label
        p_t = p * one_hot + (1 - p) * (1 - one_hot)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balance
        alpha_t = self.alpha * one_hot + (1 - self.alpha) * (1 - one_hot)

        loss = alpha_t * focal_weight * ce
        return loss.sum()


class IOULoss(nn.Module):
    """IoU Loss for bounding box regression (LTRB format)"""
    def __init__(self, loss_type="iou"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
        target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

        w_intersect = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
        h_intersect = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])
        area_intersect = w_intersect * h_intersect
        area_union = pred_area + target_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        else:
            losses = -torch.log(ious)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            return losses.sum()


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """FCOS Detection Head"""
    def __init__(self, in_channels=256, num_classes=1, num_convs=4, num_levels=4, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes

        cls_tower = []
        bbox_tower = []
        for _ in range(num_convs):
            cls_tower.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(),
            ])
            bbox_tower.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(),
            ])
        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        # Init weights
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

    def forward(self, features):
        logits, bbox_reg, centerness = [], [], []
        for level, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            bbox_feat = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_feat))
            centerness.append(self.centerness(cls_feat))
            bbox_pred = self.scales[level](self.bbox_pred(bbox_feat))
            bbox_reg.append(F.relu(bbox_pred))
        return logits, bbox_reg, centerness


class FCOSDetector(nn.Module):
    """Complete FCOS Detector with loss and inference"""
    def __init__(self, in_channels=256, num_classes=1, fpn_strides=[4, 8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.num_levels = len(fpn_strides)

        self.head = FCOSHead(in_channels, num_classes, num_levels=self.num_levels)
        self.focal_loss = SigmoidFocalLoss(gamma=2.0, alpha=0.25)
        self.iou_loss = IOULoss(loss_type='iou')
        self.centerness_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

        # Inference params
        self.score_thresh = 0.05
        self.nms_thresh = 0.6
        self.max_detections = 100

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.shape[-2:]
            stride = self.fpn_strides[level]
            shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            locations.append(
                torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1) + stride // 2
            )
        return locations

    def _compute_targets_per_level(self, locations, boxes, labels, obj_size):
        """
        locations: (M, 2)
        boxes: (G, 4) xyxy
        labels: (G,)
        obj_size: [low, high]
        Returns: labels_per_loc (M,), reg_targets_per_loc (M, 4)
        """
        M = locations.shape[0]
        G = boxes.shape[0]
        device = locations.device

        if G == 0:
            return torch.zeros(M, dtype=torch.long, device=device), \
                   torch.zeros(M, 4, dtype=torch.float32, device=device)

        xs, ys = locations[:, 0], locations[:, 1]

        # (M, G)
        l = xs[:, None] - boxes[:, 0][None]
        t = ys[:, None] - boxes[:, 1][None]
        r = boxes[:, 2][None] - xs[:, None]
        b = boxes[:, 3][None] - ys[:, None]
        reg_targets = torch.stack([l, t, r, b], dim=2)  # (M, G, 4)

        is_in_boxes = reg_targets.min(dim=2)[0] > 0  # (M, G)
        max_reg = reg_targets.max(dim=2)[0]  # (M, G)
        is_cared = (max_reg >= obj_size[0]) & (max_reg <= obj_size[1])

        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (G,)
        locations_to_gt_area = box_areas[None].expand(M, G).clone()
        locations_to_gt_area[~is_in_boxes] = 1e8
        locations_to_gt_area[~is_cared] = 1e8

        min_area, gt_inds = locations_to_gt_area.min(dim=1)  # (M,)

        reg_targets_per_loc = reg_targets[torch.arange(M, device=device), gt_inds]
        labels_per_loc = labels[gt_inds].clone()
        labels_per_loc[min_area >= 1e8] = 0

        return labels_per_loc, reg_targets_per_loc

    @staticmethod
    def compute_centerness(reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0].clamp(min=1e-6)) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0].clamp(min=1e-6))
        return torch.sqrt(centerness.clamp(min=0))

    def forward(self, features, targets=None):
        logits, bbox_reg, centerness = self.head(features)
        locations = self.compute_locations(features)

        if self.training and targets is not None:
            return self.compute_loss(logits, bbox_reg, centerness, locations, targets)
        else:
            return self.inference(logits, bbox_reg, centerness, locations, features[0].shape[0])

    def compute_loss(self, logits, bbox_reg, centerness, locations, targets):
        """
        All predictions are organized level-first:
          logits[level]: (B, C, H, W)
        We flatten as level-first, image-second to match FCOS official.
        """
        B = logits[0].shape[0]
        object_sizes = [[-1, 64], [64, 128], [128, 256], [256, 1e8]][:self.num_levels]

        # Compute targets: organized as level-first, image-second
        # i.e., for each level, concat all images' targets
        labels_level_first = []
        reg_targets_level_first = []

        for level_idx in range(self.num_levels):
            labels_per_level = []
            reg_per_level = []
            for img_idx in range(B):
                boxes = targets[img_idx]['boxes']
                gt_labels = targets[img_idx]['labels']
                lab, reg = self._compute_targets_per_level(
                    locations[level_idx], boxes, gt_labels, object_sizes[level_idx]
                )
                labels_per_level.append(lab)
                reg_per_level.append(reg)
            labels_level_first.append(torch.cat(labels_per_level))
            reg_targets_level_first.append(torch.cat(reg_per_level))

        # Flatten predictions: level-first, image-second (same order)
        logits_flat = torch.cat([
            l.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for l in logits
        ])
        bbox_reg_flat = torch.cat([
            b.permute(0, 2, 3, 1).reshape(-1, 4) for b in bbox_reg
        ])
        centerness_flat = torch.cat([c.reshape(-1) for c in centerness])

        labels_flat = torch.cat(labels_level_first)
        reg_targets_flat = torch.cat(reg_targets_level_first)

        pos_inds = labels_flat > 0
        num_pos = pos_inds.sum().clamp(min=1).float()

        cls_loss = self.focal_loss(logits_flat, labels_flat.long()) / num_pos

        if pos_inds.sum() > 0:
            ct = self.compute_centerness(reg_targets_flat[pos_inds])
            reg_loss = self.iou_loss(
                bbox_reg_flat[pos_inds], reg_targets_flat[pos_inds], ct
            ) / ct.sum().clamp(min=1)
            centerness_loss = self.centerness_loss_fn(
                centerness_flat[pos_inds], ct
            ) / num_pos
        else:
            reg_loss = bbox_reg_flat.sum() * 0
            centerness_loss = centerness_flat.sum() * 0

        return {'loss_cls': cls_loss, 'loss_reg': reg_loss, 'loss_centerness': centerness_loss}

    def inference(self, logits, bbox_reg, centerness, locations, batch_size):
        device = logits[0].device
        results = []

        for img_idx in range(batch_size):
            boxes_all = []
            scores_all = []
            labels_all = []

            for level, (logit, bbox, center, loc) in enumerate(
                zip(logits, bbox_reg, centerness, locations)
            ):
                C = logit.shape[1]
                H, W = logit.shape[2], logit.shape[3]

                scores = torch.sigmoid(logit[img_idx]).permute(1, 2, 0).reshape(-1, C)
                center_scores = torch.sigmoid(center[img_idx]).reshape(-1)
                bbox_pred = bbox[img_idx].permute(1, 2, 0).reshape(-1, 4)

                scores = scores * center_scores[:, None]
                max_scores, max_cls = scores.max(dim=1)

                keep = max_scores > self.score_thresh
                if keep.sum() == 0:
                    continue

                s = max_scores[keep]
                c = max_cls[keep] + 1
                bp = bbox_pred[keep]
                lk = loc[keep]

                boxes = torch.stack([
                    (lk[:, 0] - bp[:, 0]).clamp(min=0),
                    (lk[:, 1] - bp[:, 1]).clamp(min=0),
                    lk[:, 0] + bp[:, 2],
                    lk[:, 1] + bp[:, 3],
                ], dim=1)

                boxes_all.append(boxes)
                scores_all.append(s)
                labels_all.append(c)

            if len(boxes_all) == 0:
                results.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros(0, device=device),
                    'labels': torch.zeros(0, dtype=torch.long, device=device),
                })
                continue

            boxes_all = torch.cat(boxes_all)
            scores_all = torch.cat(scores_all)
            labels_all = torch.cat(labels_all)

            keep = nms(boxes_all, scores_all, self.nms_thresh)
            keep = keep[:self.max_detections]

            results.append({
                'boxes': boxes_all[keep],
                'scores': scores_all[keep],
                'labels': labels_all[keep],
            })

        return results
