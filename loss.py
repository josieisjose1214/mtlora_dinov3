import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tal import make_anchors, dist2bbox, bbox2dist, TaskAlignedAssigner
from models.ops import bbox_iou, xywh2xyxy


# DFL损失，YOLO检测损失
class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
            imgsz: torch.Tensor,
            stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                    F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1,
                                                                                               keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class v8DetectionLoss(nn.Module):
    """
    computing training loss
    """

    def __init__(self, tal_topk=10, tal_topk2=None):
        super(v8DetectionLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # 将stride注册为buffer，确保设备一致性
        self.register_buffer('stride', torch.tensor([8., 16., 32.]))
        self.nc = 1
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4  # reg_max*4
        # device 将从 buffer 或输入 tensor 动态获取

        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5

        self.use_dfl = self.reg_max > 1

        # 正负样本分配器,选择topk个正样本
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0,
                                            stride=self.stride.tolist(), topk2=tal_topk2)
        self.bbox_loss = BboxLoss(self.reg_max)
        # 将proj注册为buffer，这样它会被包含在state_dict中，但不会被当作参数
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float))


    def preprocess(self, targets, batch_size, scale_tensor):
        """
        预处理targets，转换成tensor格式并缩放坐标
        """
        device = targets.device  # 从输入tensor获取设备
        nl, ne = targets.shape  # nl是batch中的目标个数，ne是每个目标的元素数目4个边界框坐标和1个类别标签
        if nl == 0:
            # 无目标的情况
            out = torch.zeros(batch_size, 0, ne - 1, device=device)
        else:
            i = targets[:, 0]  # 图像index列，表示每行属于哪一个图像
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)  # counts表示每张图片上的目标数目
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)  # out用于存储每张图片的目标信息，[B,max框数目,4]
            for j in range(batch_size):
                matches = i == j  # 找到属于当前图片j的所有目标
                if n := matches.sum():  # 计算出属于当前图片的目标数n
                    out[j, :n] = targets[matches, 1:]  # 填充到out
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))  # 尺寸缩放（使其与特征图一致）和格式转换
        return out


    def bbox_decode(self, anchor_points, pred_dist):
        # 从预测的边界框坐标分布pred_dist和锚点坐标anchor_points，计算出预测框的坐标值x1y1x2y2。
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 确保proj在正确的设备上，并转换为正确的dtype
            proj = self.proj.to(pred_dist.device).type(pred_dist.dtype)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def get_assigned_targets_and_loss(self, preds, batch):
        r"""
        计算 box\cls\dfl 损失，返回前景掩码和目标索引
        """
        pred_disri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),  # [B, 4116, 64]
            preds["scores"].permute(0, 2, 1).contiguous(),  # [B, 4116, 1]
        )
        device = pred_scores.device  # 从输入tensor获取设备
        loss = torch.zeros(3, device=device)  # box, cls, dfl

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        # targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_disri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_disri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

            loss[0] *= self.box_gain
            loss[1] *= self.cls_gain
            loss[2] *= self.dfl_gain
            # loss(box, cls, dfl)
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),)


    def parse_output(self, preds):
        """
        解析模型Head的输出，如果是train就是字典,否则就是tuple
        """
        return preds[1] if isinstance(preds, tuple) else preds


    def __call__(
            self,
            preds,
            batch,
    ):
        return self.loss(self.parse_output(preds), batch)


    def loss(self, preds, batch):
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        # 返回标量损失值，与其他损失函数保持一致
        # loss是 [box_loss, cls_loss, dfl_loss] 的tensor，这些损失已经归一化（除以target_scores_sum）
        # 直接求和返回平均损失，与其他任务保持一致（不需要乘以batch_size）
        total_loss = loss.sum()  # box_loss + cls_loss + dfl_loss
        return total_loss
