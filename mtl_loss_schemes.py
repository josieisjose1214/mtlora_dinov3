# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
#
# Original file:
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Written by Simon Vandenhende
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
import math

from loss import v8DetectionLoss
from models.fcos_head import FCOSLoss


class CountLoss(Module):
    """计数任务损失：使用 PET 的 SetCriterion 与 compute_loss 逻辑。"""
    def __init__(self, num_classes=1, ce_loss_coef=1.0, point_loss_coef=1.0, eos_coef=0.4,
                 set_cost_class=1.0, set_cost_point=1.0):
        super(CountLoss, self).__init__()
        from types import SimpleNamespace
        args = SimpleNamespace(
            set_cost_class=set_cost_class,
            set_cost_point=set_cost_point,
        )
        from pet_models.matcher import build_matcher
        from pet_models.pet import SetCriterion
        self.matcher = build_matcher(args)
        weight_dict = {'loss_ce': ce_loss_coef, 'loss_points': point_loss_coef}
        losses = ['labels', 'points']
        self.criterion = SetCriterion(
            num_classes, matcher=self.matcher, weight_dict=weight_dict,
            eos_coef=eos_coef, losses=losses,
        )
        self.warmup_ep = 5

    def forward(self, outputs, targets, epoch=10):
        """
        outputs: PETCountHead 返回的 dict（sparse, dense, split_map_raw 等）
        targets: list of dict，每项为 {'points': tensor [N,2], 'labels': tensor [N], 'density': scalar}
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = self.criterion.weight_dict
        if epoch >= self.warmup_ep:
            loss_dict_sparse = self.criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = self.criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = self.criterion(output_sparse, targets)
            loss_dict_dense = self.criterion(output_dense, targets)
        loss_dict_sparse = {k + '_sp': v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k + '_sp': v for k, v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse if k in weight_dict_sparse)
        loss_dict_dense = {k + '_ds': v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k + '_ds': v for k, v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense if k in weight_dict_dense)
        losses = loss_pq_sparse + loss_pq_dense
        den = torch.tensor([t['density'].item() if t['density'].dim() == 0 else t['density'].cpu().item() for t in targets], device=outputs['split_map_raw'].device)
        bs = len(den)
        # 与 PET 头里 sparse_stride 对齐：原为 2*8=16；若 sparse_stride=4 则用 2*4=8
        sp_stride = float(outputs['sparse']['pq_stride']) if outputs['sparse'] is not None else 8.0
        ds_idx = den < (2.0 * sp_stride)
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()
        if ds_idx.sum() > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0
        loss_split = loss_split_sp + loss_split_ds
        weight_split = 0.1 if epoch >= self.warmup_ep else 0.0
        losses = losses + loss_split * weight_split
        return losses


class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self, ignore_index=255):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)

        return loss


class ClassificationLoss(Module):
    """
    Classification loss for image classification task.
    Handles 4D output [batch_size, num_classes, H, W] from decoder and 1D labels [batch_size].
    Uses global average pooling to convert 4D output to 2D [batch_size, num_classes].
    """
    
    def __init__(self, ignore_index=255):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
    
    def forward(self, out, label):
        """
        Args:
            out: Model output [batch_size, num_classes, H, W] or [batch_size, num_classes]
            label: Ground truth labels [batch_size] or [batch_size, 1]
        """
        assert not label.requires_grad
        
        # 处理label格式：确保是1D张量
        if label.dim() > 1:
            label = label.squeeze().long()
        else:
            label = label.long()
        
        # 处理out格式：如果是4D，使用全局平均池化转换为2D
        if out.dim() == 4:
            # [batch_size, num_classes, H, W] -> [batch_size, num_classes]
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), out.size(1))  # [batch_size, num_classes]
        elif out.dim() == 3:
            # [batch_size, num_classes, L] -> [batch_size, num_classes]
            out = out.mean(dim=2)
        elif out.dim() == 2:
            # 已经是2D [batch_size, num_classes]
            pass
        else:
            raise ValueError(f"Unexpected output shape: {out.shape}")
        
        # 确保out是2D [batch_size, num_classes]
        assert out.dim() == 2, f"Expected 2D output after processing, got {out.dim()}D"
        assert label.dim() == 1, f"Expected 1D label, got {label.dim()}D"
        
        # 过滤掉ignore_index的样本
        if self.ignore_index is not None:
            valid_mask = (label != self.ignore_index)
            if valid_mask.sum() == 0:
                # 如果没有有效样本，返回零损失
                return torch.tensor(0.0, device=out.device, requires_grad=True)
            out = out[valid_mask]
            label = label[valid_mask]
        
        loss = self.criterion(out, label)
        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """

    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError(
                'Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        mask = (label != 255)
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """

    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            # print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            # print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(
                out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(
                out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss


class SingleTaskLoss(nn.Module):
    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    def forward(self, pred, gt):
        out = {self.task: self.loss_ft(pred[self.task], gt[self.task])}
        out['total'] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {
            task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks
        }
        out['total'] = torch.sum(torch.stack(
            [self.loss_weights[t] * out[t] for t in self.tasks]))
        return out['total'], out


def get_loss(task_cfg, task=None, config={"DATA": {}}):
    """ Return loss function for a specific task """
    if task == 'edge':
        criterion = BalancedCrossEntropyLoss(
            size_average=True, pos_weight=task_cfg.get('edge_w', 0.95))

    elif task == 'semseg' or task == 'human_parts':
        criterion = SoftMaxwithLoss(ignore_index=255)

    elif task == 'classify':
        # 分类任务：模型输出是4D [B, C, H, W]，但标签是1D [B]
        # 使用自定义损失函数来处理这种情况
        criterion = ClassificationLoss(ignore_index=255)

    elif task == 'normals':
        criterion = NormalsLoss(normalize=True, size_average=True, norm=1)

    elif task == 'sal':
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        criterion = DepthLoss('l1')

    elif task == 'count':
        criterion = CountLoss(num_classes=1)

    elif task == 'detect':
        detect_head_type = None
        if hasattr(config, 'MODEL') and hasattr(config.MODEL, 'DECODER_HEAD'):
            detect_head_type = config.MODEL.DECODER_HEAD.get('detect', None)
        if detect_head_type == 'fcos':
            num_classes = 1
            try:
                num_classes = int(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT.get('detect', 1))
            except Exception:
                pass
            criterion = FCOSLoss(num_classes=num_classes, num_levels=4)
        else:
            criterion = v8DetectionLoss(tal_topk=10)
        # criterion = DetectionLoss(
        #     lambda_coord=task_cfg.get('lambda_coord', 1.0),
        #     lambda_cls=task_cfg.get('lambda_cls', 1.0),
        #     lambda_pos=task_cfg.get('lambda_pos', 0.0),  # 位置损失权重（默认0，CIoU已包含位置信息）
        #     use_smooth_l1=task_cfg.get('use_smooth_l1', False),  # 是否使用Smooth L1损失
        #     smooth_l1_beta=task_cfg.get('smooth_l1_beta', 1.0),  # Smooth L1的beta参数
        #     ignore_index=task_cfg.get('ignore_index', -1),
        #     box_format=task_cfg.get('box_format', 'xywh'),  # 默认使用xywh格式
        #     num_classes=task_cfg.get('num_classes', 1),  # 默认1类（小麦检测）
        #     iou_threshold=task_cfg.get('iou_threshold', 0.5),  # IoU匹配阈值
        #     reg_max=task_cfg.get('reg_max', 16)  # DFL的reg_max参数（YOLOHead使用）
        # )

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, classify, detect, count, or normals')

    return criterion
