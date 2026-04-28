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
import numpy as np
from collections import defaultdict


class DetectionMeter(object):
    """ Performance meter for object detection tasks """
    
    def __init__(self, num_classes=1, iou_threshold=0.5, conf_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.reset()
    
    def reset(self):
        self.predictions = []  # List of [x1, y1, x2, y2, conf, class]
        self.targets = []      # List of [x1, y1, x2, y2, class]
        self.image_ids = []    # Track which predictions/targets belong to which image
        self.loss = 0.0
        self.n_samples = 0
        self.image_counter = 0
        # Debug counters (must exist before first update call)
        self._debug_print_count = 0
        self._iou_debug_count = 0
    
    def _convert_format(self, boxes, format='xywh'):
        """Convert box format from xywh to xyxy"""
        if format == 'xywh':
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return np.stack([x1, y1, x2, y2], axis=1)
        return boxes
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in xyxy format"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    @staticmethod
    def _compute_ap(recalls, precisions):
        recalls = np.concatenate([[0.0], recalls, [1.0]])
        precisions = np.concatenate([[0.0], precisions, [0.0]])
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return float(ap)
    
    def _calculate_map(self, predictions, targets, image_ids, iou_threshold=None):
        """Calculate mAP at a given IoU threshold."""
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        if len(predictions) == 0:
            return {'mAP': 0.0, 'mAP_per_class': np.zeros(self.num_classes)}
        
        # Group by class
        pred_by_class = defaultdict(list)
        target_by_class = defaultdict(list)
        
        for pred in predictions:
            pred_by_class[int(pred[5])].append(pred)
        
        for target in targets:
            target_by_class[int(target[4])].append(target)
        
        # Calculate AP for each class
        ap_per_class = []
        
        for class_id in range(self.num_classes):
            class_preds = pred_by_class[class_id]
            class_targets = target_by_class[class_id]
            
            if len(class_targets) == 0:
                ap_per_class.append(0.0)
                continue
            
            if len(class_preds) == 0:
                ap_per_class.append(0.0)
                continue
            
            # Sort predictions by confidence
            class_preds = sorted(class_preds, key=lambda x: x[4], reverse=True)
            
            # Calculate precision and recall
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            
            # Track which targets have been matched
            matched_targets = set()
            
            # Group targets by image_id for faster lookup
            targets_by_image = defaultdict(list)
            for target_idx, target in enumerate(class_targets):
                targets_by_image[target[5]].append((target_idx, target[:4]))
            
            for pred_idx, pred in enumerate(class_preds):
                pred_box = pred[:4]
                pred_image_id = pred[6]
                
                # Find matching targets in the same image (only check relevant targets)
                best_iou = 0
                best_target_idx = -1
                
                # Only check targets in the same image
                if pred_image_id in targets_by_image:
                    for target_idx, target_box in targets_by_image[pred_image_id]:
                        iou = self._calculate_iou(pred_box, target_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = target_idx
                
                # 调试信息：检查IoU匹配情况（只打印前几个）
                if not hasattr(self, '_iou_debug_count'):
                    self._iou_debug_count = 0
                self._iou_debug_count += 1
                if self._iou_debug_count <= 0 and class_id == 0:
                    if pred_image_id in targets_by_image and len(targets_by_image[pred_image_id]) > 0:
                        print(f"\n[IoU DEBUG] Pred box: {pred_box}, Best IoU: {best_iou:.4f}, Threshold: {iou_threshold:.4f}")
                        if best_iou > 0 and best_target_idx >= 0:
                            # best_target_idx是class_targets中的索引，需要找到对应的target_box
                            # 由于targets_by_image存储的是(target_idx, target_box)，我们需要找到匹配的
                            matched_target_box = None
                            for t_idx, t_box in targets_by_image[pred_image_id]:
                                if t_idx == best_target_idx:
                                    matched_target_box = t_box
                                    break
                            if matched_target_box is not None:
                                print(f"[IoU DEBUG] Matched target: {matched_target_box}")
                
                if best_iou >= iou_threshold and best_target_idx not in matched_targets:
                    tp[pred_idx] = 1
                    matched_targets.add(best_target_idx)
                else:
                    fp[pred_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(class_targets)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            
            ap = self._compute_ap(recalls, precisions)
            
            # 调试信息：打印匹配统计（只打印第一个类别）
            if class_id == 0 and not hasattr(self, '_ap_debug_printed') and False:
                self._ap_debug_printed = True
                print(f"\n[AP DEBUG] Class {class_id}:")
                print(f"  Total predictions: {len(class_preds)}")
                print(f"  Total targets: {len(class_targets)}")
                print(f"  True positives: {tp.sum()}")
                print(f"  False positives: {fp.sum()}")
                print(f"  Precision: {precisions[-1]:.4f} (at recall {recalls[-1]:.4f})")
                print(f"  Max precision: {precisions.max():.4f}")
                print(f"  Max recall: {recalls.max():.4f}")
                print(f"  AP: {ap:.4f}")
            
            ap_per_class.append(ap)
        
        mAP = np.mean(ap_per_class)
        
        return {
            'mAP': mAP,
            'mAP_per_class': np.array(ap_per_class),
            'AP': ap_per_class
        }
    
    @torch.no_grad()
    def update(self, pred, gt, image_size=(448, 448)):
        """
        Update meter with predictions and ground truth
        
        Args:
            pred: Model predictions - can be:
                  - Tensor of shape [B, N, 6] where N is max detections per image, 
                    each detection is [x, y, w, h, conf, class]
                  - List of tensors with variable number of detections per image
            gt: Ground truth boxes - can be:
                - Tensor of shape [B, M, 5] where M is max GT boxes per image,
                  each box is [class, x, y, w, h] in relative coordinates
                - List of tensors with variable number of GT boxes per image
            image_size: Tuple of (height, width) for converting relative to absolute coordinates
        """
        batch_size = 1  # Default for single image
        
        # Handle different prediction formats
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 3:  # [B, N, 6]
                batch_size = pred.size(0)
                pred_list = [pred[i] for i in range(batch_size)]
            elif pred.dim() == 2:  # [N, 6]
                pred_list = [pred]
            else:
                raise ValueError(f"Unexpected prediction tensor shape: {pred.shape}")
        elif isinstance(pred, list):
            pred_list = pred
            batch_size = len(pred_list)  # Use list length as batch size
        else:
            raise ValueError(f"Unexpected prediction type: {type(pred)}")
        
        # Handle different ground truth formats
        if isinstance(gt, torch.Tensor):
            if gt.dim() == 3:  # [B, M, 5]
                batch_size = max(batch_size, gt.size(0))  # Take max in case pred and gt have different batch sizes
                gt_list = [gt[i] for i in range(gt.size(0))]
            elif gt.dim() == 2:  # [M, 5]
                gt_list = [gt]
            else:
                raise ValueError(f"Unexpected ground truth tensor shape: {gt.shape}")
        elif isinstance(gt, list):
            gt_list = gt
            batch_size = max(batch_size, len(gt_list))  # Take max in case pred and gt have different lengths
        else:
            raise ValueError(f"Unexpected ground truth type: {type(gt)}")
        
        # Process each image in the batch
        for batch_idx in range(batch_size):
            current_pred = pred_list[batch_idx] if batch_idx < len(pred_list) else torch.empty(0, 6)
            current_gt = gt_list[batch_idx] if batch_idx < len(gt_list) else torch.empty(0, 5)
            
            # Filter predictions by confidence threshold
            if current_pred.size(0) > 0:
                valid_mask = current_pred[:, 4] >= self.conf_threshold
                current_pred = current_pred[valid_mask]
            
            # Convert predictions to numpy and add image ID (batch convert to reduce overhead)
            if current_pred.size(0) > 0:
                # NMS输出格式是 [x1, y1, x2, y2, conf, class] (xyxy格式，绝对坐标)
                # 直接提取xyxy坐标，不需要转换
                pred_boxes_np = current_pred[:, :4].cpu().numpy()  # [x1, y1, x2, y2] (xyxy格式)
                pred_conf_np = current_pred[:, 4].cpu().numpy()
                pred_class_np = current_pred[:, 5].cpu().numpy()
                
                # 调试信息：检查DetectionMeter接收到的预测框格式
                if not hasattr(self, '_debug_print_count'):
                    self._debug_print_count = 0
                self._debug_print_count += 1
                if self._debug_print_count <= 0:
                    print(f"\n[DetectionMeter.update DEBUG] current_pred shape: {current_pred.shape}")
                    print(f"[DetectionMeter.update DEBUG] pred_boxes_np shape: {pred_boxes_np.shape}")
                    print(f"[DetectionMeter.update DEBUG] pred_boxes_np range: min={pred_boxes_np.min():.4f}, max={pred_boxes_np.max():.4f}")
                    print(f"[DetectionMeter.update DEBUG] pred_boxes_np sample (first 3): {pred_boxes_np[:min(3, len(pred_boxes_np)), :]}")
                    print(f"[DetectionMeter.update DEBUG] pred_boxes_np coord ranges: x1=[{pred_boxes_np[:, 0].min():.4f}, {pred_boxes_np[:, 0].max():.4f}], "
                          f"y1=[{pred_boxes_np[:, 1].min():.4f}, {pred_boxes_np[:, 1].max():.4f}], "
                          f"x2=[{pred_boxes_np[:, 2].min():.4f}, {pred_boxes_np[:, 2].max():.4f}], "
                          f"y2=[{pred_boxes_np[:, 3].min():.4f}, {pred_boxes_np[:, 3].max():.4f}]")
                
                # NMS输出已经是xyxy格式的绝对坐标，直接使用
                # 但需要确保坐标在图像范围内（裁剪到图像边界）
                h, w = image_size
                x1 = np.clip(pred_boxes_np[:, 0], 0, w)
                y1 = np.clip(pred_boxes_np[:, 1], 0, h)
                x2 = np.clip(pred_boxes_np[:, 2], 0, w)
                y2 = np.clip(pred_boxes_np[:, 3], 0, h)
                
                # 过滤掉无效的框（x2 <= x1 或 y2 <= y1）
                valid_mask = (x2 > x1) & (y2 > y1)
                x1 = x1[valid_mask]
                y1 = y1[valid_mask]
                x2 = x2[valid_mask]
                y2 = y2[valid_mask]
                pred_conf_np = pred_conf_np[valid_mask]
                pred_class_np = pred_class_np[valid_mask]
                
                # 调试信息：检查裁剪后的坐标
                if self._debug_print_count <= 0 and len(x1) > 0:
                    print(f"[DetectionMeter.update DEBUG] After clipping - x1 range: [{x1.min():.1f}, {x1.max():.1f}], "
                          f"y1 range: [{y1.min():.1f}, {y1.max():.1f}], "
                          f"x2 range: [{x2.min():.1f}, {x2.max():.1f}], "
                          f"y2 range: [{y2.min():.1f}, {y2.max():.1f}]")
                    print(f"[DetectionMeter.update DEBUG] Valid boxes: {len(x1)}/{len(pred_boxes_np)}")
                    if len(x1) > 0:
                        print(f"[DetectionMeter.update DEBUG] Sample box (first): x1={x1[0]:.1f}, y1={y1[0]:.1f}, x2={x2[0]:.1f}, y2={y2[0]:.1f}")
                
                # Batch append (more efficient than individual appends)
                image_id = self.image_counter
                for idx in range(len(x1)):
                    self.predictions.append([
                        x1[idx], y1[idx], x2[idx], y2[idx],
                        pred_conf_np[idx], pred_class_np[idx], image_id
                    ])
            
            # Convert ground truth to numpy (batch convert)
            if current_gt.size(0) > 0:
                # Batch convert to numpy once
                gt_boxes_np = current_gt[:, 1:5].cpu().numpy()  # [x, y, w, h]
                gt_class_np = current_gt[:, 0].cpu().numpy()
                
                # 调试信息：检查DetectionMeter接收到的真实框格式
                if self._debug_print_count <= 0:
                    print(f"[DetectionMeter.update DEBUG] current_gt shape: {current_gt.shape}")
                    print(f"[DetectionMeter.update DEBUG] gt_boxes_np shape: {gt_boxes_np.shape}")
                    print(f"[DetectionMeter.update DEBUG] gt_boxes_np range: min={gt_boxes_np.min():.4f}, max={gt_boxes_np.max():.4f}")
                    print(f"[DetectionMeter.update DEBUG] gt_boxes_np sample (first 3): {gt_boxes_np[:min(3, len(gt_boxes_np)), :]}")
                    print(f"[DetectionMeter.update DEBUG] gt_boxes_np coord ranges: x=[{gt_boxes_np[:, 0].min():.4f}, {gt_boxes_np[:, 0].max():.4f}], "
                          f"y=[{gt_boxes_np[:, 1].min():.4f}, {gt_boxes_np[:, 1].max():.4f}], "
                          f"w=[{gt_boxes_np[:, 2].min():.4f}, {gt_boxes_np[:, 2].max():.4f}], "
                          f"h=[{gt_boxes_np[:, 3].min():.4f}, {gt_boxes_np[:, 3].max():.4f}]")
                
                # Check if coordinates are already in absolute format (range > 1.5) or relative (range <= 1.5)
                max_coord = gt_boxes_np.max()
                min_coord = gt_boxes_np.min()
                # 更严格的判断：如果坐标范围明显大于1.5，或者中心点坐标大于1.5，认为是绝对坐标
                # 同时检查宽高：如果宽高明显大于1.5，也认为是绝对坐标
                is_absolute = (max_coord > 1.5) or (gt_boxes_np[:, 0].max() > 1.5) or (gt_boxes_np[:, 1].max() > 1.5) or \
                             (gt_boxes_np[:, 2].max() > 1.5) or (gt_boxes_np[:, 3].max() > 1.5)
                
                if self._debug_print_count <= 0:
                    print(f"[DetectionMeter.update DEBUG] GT is_absolute: {is_absolute}, max_coord: {max_coord:.4f}, min_coord: {min_coord:.4f}")
                
                # Convert relative coordinates to absolute (vectorized)
                x_center, y_center, width, height = gt_boxes_np[:, 0], gt_boxes_np[:, 1], gt_boxes_np[:, 2], gt_boxes_np[:, 3]
                
                if is_absolute:
                    # Coordinates are already absolute, just convert from center+size to x1y1x2y2
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                else:
                    # Coordinates are relative (0-1), convert to absolute
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # Scale to image size
                    h, w = image_size
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
                
                # 调试信息：检查转换后的真实框坐标
                if self._debug_print_count <= 0:
                    print(f"[DetectionMeter.update DEBUG] GT After conversion - x1 range: [{x1.min():.1f}, {x1.max():.1f}], "
                          f"y1 range: [{y1.min():.1f}, {y1.max():.1f}], "
                          f"x2 range: [{x2.min():.1f}, {x2.max():.1f}], "
                          f"y2 range: [{y2.min():.1f}, {y2.max():.1f}]")
                    if len(x1) > 0:
                        print(f"[DetectionMeter.update DEBUG] Sample converted GT box (first): x1={x1[0]:.1f}, y1={y1[0]:.1f}, x2={x2[0]:.1f}, y2={y2[0]:.1f}")
                
                # Batch append
                image_id = self.image_counter
                for idx in range(len(gt_boxes_np)):
                    self.targets.append([
                        x1[idx], y1[idx], x2[idx], y2[idx],
                        gt_class_np[idx], image_id
                    ])
            
            self.image_counter += 1
        
        self.n_samples += batch_size
    
    def update_loss(self, loss, n_samples):
        """Update running loss"""
        self.loss += loss * n_samples
        self.n_samples += n_samples
    
    def get_score(self, verbose=True):
        """
        Calculate and return detection metrics
        
        Returns:
            dict: Dictionary containing mAP and per-class AP
        """
        if len(self.targets) == 0:
            if verbose:
                print('\nDetection Evaluation')
                print(f'Warning: No ground truth targets found. Returning default metrics.')
                print(f'Number of images processed: {self.n_samples}')
                print(f'Number of predictions: {len(self.predictions)}')
            return {
                'mAP': 0.0,
                'map': 0.0,  # 兼容 main 中的 eval_results['detect']['map']
                'mAP_per_class': np.zeros(self.num_classes),
                'n_images': self.n_samples,
                'n_predictions': len(self.predictions),
                'n_targets': 0
            }
        
        if len(self.predictions) == 0:
            if verbose:
                print('\nDetection Evaluation')
                print(f'Warning: No predictions found (all filtered by confidence threshold).')
                print(f'Number of images: {self.n_samples}')
                print(f'Number of targets: {len(self.targets)}')
            return {
                'mAP': 0.0,
                'map': 0.0,  # 兼容 main 中的 eval_results['detect']['map']
                'mAP_per_class': np.zeros(self.num_classes),
                'n_images': self.n_samples,
                'n_predictions': 0,
                'n_targets': len(self.targets)
            }
        
        # Calculate AP50/AP75 and averaged AP (same as single_task/main_detect.py)
        try:
            map50 = self._calculate_map(self.predictions, self.targets, self.image_ids, iou_threshold=0.5)
            map75 = self._calculate_map(self.predictions, self.targets, self.image_ids, iou_threshold=0.75)
        except Exception as e:
            if verbose:
                print(f'\nDetection Evaluation Error: {e}')
                print(f'Number of predictions: {len(self.predictions)}')
                print(f'Number of targets: {len(self.targets)}')
            return {
                'mAP': 0.0,
                'map': 0.0,  # 兼容 main 中的 eval_results['detect']['map']
                'mAP_per_class': np.zeros(self.num_classes),
                'n_images': self.n_samples,
                'n_predictions': len(self.predictions),
                'n_targets': len(self.targets),
                'error': str(e)
            }
        
        ap50 = float(map50['mAP'])
        ap75 = float(map75['mAP'])
        ap = (ap50 + ap75) / 2.0
        eval_dict = {
            'AP50': ap50,
            'AP75': ap75,
            'AP': ap,
            'mAP': ap50,  # keep compatibility; old mAP means AP50 here
            'map': ap,    # main.py historically reads `map`; now aligns with single_task AP
            'mAP_per_class': map50['mAP_per_class'],
            'n_images': self.n_samples,
            'n_predictions': len(self.predictions),
            'n_targets': len(self.targets)
        }
        
        if self.loss > 0:
            eval_dict['loss'] = self.loss / max(self.n_samples, 1)
        
        if verbose:
            print('\nDetection Evaluation')
            print(f'AP50: {eval_dict["AP50"]:.4f}')
            print(f'AP75: {eval_dict["AP75"]:.4f}')
            print(f'AP: {eval_dict["AP"]:.4f}')
            print(f'Number of images: {eval_dict["n_images"]}')
            print(f'Number of predictions: {eval_dict["n_predictions"]}')
            print(f'Number of targets: {eval_dict["n_targets"]}')
            print(f'Average predictions per image: {eval_dict["n_predictions"] / max(eval_dict["n_images"], 1):.2f}')
            print(f'Average targets per image: {eval_dict["n_targets"] / max(eval_dict["n_images"], 1):.2f}')
            
            # Print some statistics about predictions
            if len(self.predictions) > 0:
                pred_confs = [p[4] for p in self.predictions]
                print(f'Prediction confidence range: [{min(pred_confs):.4f}, {max(pred_confs):.4f}], mean: {np.mean(pred_confs):.4f}')
                
                # Check if predictions are in valid coordinate range
                pred_x1 = [p[0] for p in self.predictions]
                pred_y1 = [p[1] for p in self.predictions]
                pred_x2 = [p[2] for p in self.predictions]
                pred_y2 = [p[3] for p in self.predictions]
                
                # Calculate box sizes for debugging
                pred_widths = [p[2] - p[0] for p in self.predictions]
                pred_heights = [p[3] - p[1] for p in self.predictions]
                
                print(f'Prediction box coordinates - x: [{min(pred_x1):.1f}, {max(pred_x2):.1f}], y: [{min(pred_y1):.1f}, {max(pred_y2):.1f}]')
                print(f'Prediction box sizes - width: [{min(pred_widths):.1f}, {max(pred_widths):.1f}], height: [{min(pred_heights):.1f}, {max(pred_heights):.1f}]')
                
                # Also print GT box statistics for comparison
                if len(self.targets) > 0:
                    gt_x1 = [t[0] for t in self.targets]
                    gt_y1 = [t[1] for t in self.targets]
                    gt_x2 = [t[2] for t in self.targets]
                    gt_y2 = [t[3] for t in self.targets]
                    gt_widths = [t[2] - t[0] for t in self.targets]
                    gt_heights = [t[3] - t[1] for t in self.targets]
                    print(f'Ground truth box coordinates - x: [{min(gt_x1):.1f}, {max(gt_x2):.1f}], y: [{min(gt_y1):.1f}, {max(gt_y2):.1f}]')
                    print(f'Ground truth box sizes - width: [{min(gt_widths):.1f}, {max(gt_widths):.1f}], height: [{min(gt_heights):.1f}, {max(gt_heights):.1f}]')
            
            # 打印每个类别在 IoU=0.5 下的 AP
            per_class_ap50 = eval_dict.get('mAP_per_class', [])
            if len(per_class_ap50) > 0:
                print('\nPer-class AP:')
                for i in range(len(per_class_ap50)):
                    print(f'Class {i}: {per_class_ap50[i]:.4f}')
            
            if self.loss > 0:
                print(f'Average Loss: {eval_dict["loss"]:.4f}')
        
        return eval_dict