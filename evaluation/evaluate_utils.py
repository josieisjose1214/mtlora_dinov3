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

import os
import numpy as np
import torch
import torch.nn.functional as F
from .nms import non_max_suppression
from models.fcos_head import fcos_inference

def get_output(output, task, image_size=448):
    """
    Process model output for different tasks
    
    Args:
        output: Model output tensor
        task: Task name
        image_size: Image size for coordinate normalization (default: 448)
    
    Returns:
        Processed output suitable for evaluation
    """
    if task in {'normals', 'semseg', 'human_parts', 'edge', 'sal', 'depth'}:
        # Tasks that output feature maps
        output = output.permute(0, 2, 3, 1)
        
        if task == 'normals':
            output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0
        
        elif task in {'semseg', 'human_parts'}:
            _, output = torch.max(output, dim=3)
        
        elif task in {'edge', 'sal'}:
            output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
        
        elif task in {'depth'}:
            pass
    
    elif task == 'classify':
        # Classification: output is already in correct format [B, num_classes]
        pass

    elif task == 'count':
        # PET 计数：优先使用按单图统计的预测点数
        if isinstance(output, dict) and 'pred_count_per_image' in output:
            output = output['pred_count_per_image'].detach()
        elif isinstance(output, dict) and 'pred_points' in output:
            # 兼容旧逻辑：只有 batch 聚合点数时，退化为单元素张量
            n_pred = output['pred_points'].shape[1]
            output = torch.tensor([float(n_pred)], dtype=torch.float32, device=output['pred_points'].device)
        else:
            output = torch.tensor([0.0])
    
    elif task == 'detect':
        # Detection: 处理多种输出格式
        # YOLOHead测试输出：元组 (y, preds)或y, y为[B, xywh+nc, 4116]，y还需要NMS
        # 处理YOLOHead的测试输出格式（元组或已解码的检测框）
        if isinstance(output, dict) and {'logits', 'bbox_reg', 'centerness', 'locations'}.issubset(output.keys()):
            output = fcos_inference(output, score_thresh=0.5, nms_thresh=0.6, max_detections=100)
        if isinstance(output, tuple):
            # 测试时返回 (y, preds)，y 是检测框，preds 是原始预测字典
            y, preds = output
            # 提高置信度阈值，减少低质量预测框数量
            det = non_max_suppression(y, conf_thres=0.5, iou_thres=0.5, nc=1)
            #det: list[Tensor], len = batch_size; det[i].shape = (Ni, 6)
            output=det
        # 处理直接是张量的情况（非元组）
        elif isinstance(output, torch.Tensor) and output.dim() == 3:
            # [B, N, K] 格式，直接处理
            batch_size = output.size(0)
            device = output.device
            
            detections_list = []
            for b in range(batch_size):
                detections_b = output[b]  # [N, K]
                
                if detections_b.size(0) == 0:
                    detections_list.append(torch.empty(0, 6, device=device))
                    continue
                
                # 处理格式（与元组处理相同）
                if detections_b.size(1) == 6:
                    detections = detections_b
                elif detections_b.size(1) == 5:
                    last_col = detections_b[:, 4]
                    if last_col.max() <= 1.0 and last_col.min() >= 0.0:
                        pred_class = torch.zeros(detections_b.size(0), dtype=torch.long, device=device)
                        detections = torch.stack([
                            detections_b[:, 0], detections_b[:, 1], detections_b[:, 2],
                            detections_b[:, 3], detections_b[:, 4], pred_class.float()
                        ], dim=1)
                    else:
                        pred_conf = torch.ones(detections_b.size(0), device=device)
                        detections = torch.stack([
                            detections_b[:, 0], detections_b[:, 1], detections_b[:, 2],
                            detections_b[:, 3], pred_conf, detections_b[:, 4].float()
                        ], dim=1)
                else:
                    detections = torch.cat([
                        detections_b[:, :4],
                        torch.ones(detections_b.size(0), 1, device=device),
                        torch.zeros(detections_b.size(0), 1, device=device)
                    ], dim=1)
                
                # 确保坐标在合理范围内
                detections[:, 0] = torch.clamp(detections[:, 0], min=0.0, max=1.0)
                detections[:, 1] = torch.clamp(detections[:, 1], min=0.0, max=1.0)
                detections[:, 2] = torch.clamp(detections[:, 2], min=0.001)
                detections[:, 3] = torch.clamp(detections[:, 3], min=0.001)
                detections[:, 4] = torch.clamp(detections[:, 4], min=0.0, max=1.0)
                
                # 过滤低置信度
                conf_threshold = 0.5
                valid_mask = detections[:, 4] >= conf_threshold
                if valid_mask.sum() > 0:
                    detections_list.append(detections[valid_mask])
                else:
                    detections_list.append(torch.empty(0, 6, device=device))
            
            output = detections_list
    else:
        raise ValueError(f'Unknown task: {task}. Select one of the valid tasks')

    return output


class CountMeter(object):
    """计数任务评估：MAE (平均绝对误差)。"""
    def __init__(self):
        self.pred_counts = []
        self.gt_counts = []

    def reset(self):
        self.pred_counts = []
        self.gt_counts = []

    def update(self, pred, gt, **kwargs):
        # pred: tensor/list/scalar，可为单图计数或 batch 内每图计数
        # gt: list of tensor 或 list of int，每张图的 gt 点数
        if torch.is_tensor(pred):
            pred_list = pred.detach().float().cpu().reshape(-1).tolist()
        elif isinstance(pred, (list, tuple, np.ndarray)):
            pred_list = [float(x) for x in pred]
        else:
            pred_list = [float(pred)]

        def _count_points(g):
            # g 期望是 Nx2（或 2xN、或扁平 2N）；返回点数 N
            if torch.is_tensor(g):
                if g.numel() == 0:
                    return 0
                if g.dim() == 2 and g.size(-1) == 2:
                    return int(g.size(0))
                if g.dim() == 2 and g.size(0) == 2 and g.size(1) != 2:
                    return int(g.size(1))
                if g.dim() == 1:
                    return int(g.numel() // 2)
                # 兜底：尽量按 2 个数一个点
                return int(g.numel() // 2)
            # list/np 等
            try:
                n = len(g)
            except Exception:
                return 0
            # 若是扁平 2N，则点数为 n//2；否则按 n（例如 list of (x,y)）
            return int(n // 2) if (n > 0 and n % 2 == 0) else int(n)

        if isinstance(gt, (list, tuple)):
            gt_list = [_count_points(g) for g in gt]
            if len(pred_list) == len(gt_list):
                # 标准 per-image 评估：一一对应后逐图累计
                self.pred_counts.extend(pred_list)
                self.gt_counts.extend([float(x) for x in gt_list])
            else:
                # 兼容旧输出：pred 是 batch 聚合值
                gt_total = float(sum(gt_list)) if gt_list else 0.0
                self.pred_counts.append(float(pred_list[0]) if pred_list else 0.0)
                self.gt_counts.append(gt_total)
        else:
            self.pred_counts.append(float(pred_list[0]) if pred_list else 0.0)
            self.gt_counts.append(float(gt))

    def get_score(self, verbose=True):
        if not self.pred_counts:
            return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'count': 0}
        pred = np.array(self.pred_counts)
        gt = np.array(self.gt_counts)
        mae = np.abs(pred - gt).mean()
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        mean_pred = float(pred.mean()) if len(pred) else 0.0
        mean_gt = float(gt.mean()) if len(gt) else 0.0
        sst = np.sum((gt - mean_gt) ** 2)
        if len(gt) >= 2 and sst > 1e-12:
            r2 = float(1.0 - np.sum((gt - pred) ** 2) / sst)
        else:
            r2 = 0.0
        if verbose:
            print(f"  [count] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return {'mae': float(mae), 'rmse': rmse, 'r2': r2, 'count': len(pred), 'mean_pred': mean_pred, 'mean_gt': mean_gt}


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """

    def __init__(self, config, db_name="NYUD"):
        self.database = db_name
        self.tasks = config.TASKS
        self.meters = {t: get_single_task_meter(config,
                                                t, self.database) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert (set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']:  # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']

        elif task == 'normals':  # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean'])/stl['mean']

        elif task == 'edge':  # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF'])/stl['odsF']
        
        elif task == 'classify':  # accuracy higher is better
            mtl_performance += (mtl['accuracy'] - stl['accuracy'])/stl['accuracy']
        
        elif task == 'detect':  # mAP higher is better
            mtl_performance += (mtl['mAP'] - stl['mAP'])/stl['mAP']

        elif task == 'count':  # MAE lower is better
            stl_mae = max(stl.get('mae', 0.0), 1e-6)
            mtl_performance += (stl_mae - mtl.get('mae', stl_mae)) / stl_mae

        else:
            raise NotImplementedError(f"Unknown task: {task}")

    return mtl_performance / num_tasks

# TODO change database to handle more datasets


def get_single_task_meter(config, task, database="NYUD"):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database, config)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter()

    # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
    elif task == 'edge':
        from evaluation.eval_edge import EdgeMeter
        # TODO: get edge_w from task config
        return EdgeMeter(pos_weight=0.95)
        # return EdgeMeter()
    
    elif task == 'classify':
        from evaluation.eval_classify import ClassificationMeter
        # Get number of classes from config
        num_classes = 8
        try:
            num_classes = int(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT.get('classify', 8))
        except Exception:
            pass
        return ClassificationMeter(num_classes=num_classes)
    
    elif task == 'detect':
        from evaluation.eval_detect import DetectionMeter
        # Get detection parameters from config
        num_classes = 1
        eval_iou_threshold = 0.5
        return DetectionMeter(
            num_classes=num_classes,
            iou_threshold=eval_iou_threshold,
            conf_threshold=0.5
        )

    elif task == 'count':
        return CountMeter()

    else:
        raise NotImplementedError(f"Unknown task: {task}")
