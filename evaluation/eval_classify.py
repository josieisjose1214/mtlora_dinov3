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
from sklearn.metrics import accuracy_score


class ClassificationMeter(object):
    """ Performance meter for classification tasks """
    
    def __init__(self, num_classes=8):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probs = []
        self.loss = 0.0
        self.n_samples = 0
    
    @torch.no_grad()
    def update(self, pred, gt):
        """
        Update meter with predictions and ground truth
        
        Args:
            pred: Model predictions (logits or probabilities), shape [B, num_classes]
            gt: Ground truth labels, shape [B] or [B, 1]
        """
        if pred.dim() == 2 and pred.size(1) == self.num_classes:
            probs = torch.softmax(pred, dim=1).detach().cpu().numpy()
            pred_labels = np.argmax(probs, axis=1)
            self.probs.extend(probs.tolist())
        elif pred.dim() == 1:
            pred_labels = pred.detach().cpu().numpy()
            one_hot = np.zeros((len(pred_labels), self.num_classes), dtype=np.float32)
            for i, cls in enumerate(pred_labels):
                if 0 <= int(cls) < self.num_classes:
                    one_hot[i, int(cls)] = 1.0
            self.probs.extend(one_hot.tolist())
        else:
            raise ValueError(f"Unexpected prediction shape: {pred.shape}")
        
        # Handle ground truth format
        if gt.dim() == 2 and gt.size(1) == 1:
            gt_labels = gt.squeeze(1).cpu().numpy()
        elif gt.dim() == 1:
            gt_labels = gt.cpu().numpy()
        else:
            raise ValueError(f"Unexpected ground truth shape: {gt.shape}")
        
        self.predictions.extend(pred_labels.tolist())
        self.targets.extend(gt_labels.tolist())
        self.n_samples += len(pred_labels)
    
    def update_loss(self, loss, n_samples):
        """Update running loss"""
        self.loss += loss * n_samples
        self.n_samples += n_samples
    
    def get_score(self, verbose=True):
        """
        Calculate and return classification metrics
        
        Returns:
            dict: Dictionary containing accuracy, precision, recall, f1-score
        """
        if len(self.predictions) == 0:
            return {'accuracy': 0.0, 'ba': 0.0, 'mAP': 0.0, 'acc': 0.0, 'loss': 0.0}
        
        y_pred = np.array(self.predictions)
        y_true = np.array(self.targets)
        y_probs = np.array(self.probs) if len(self.probs) else np.zeros((len(y_true), self.num_classes), dtype=np.float32)
        
        accuracy = accuracy_score(y_true, y_pred)

        # Balanced Accuracy: mean recall per present class
        per_class_recall = []
        for c in range(self.num_classes):
            mask = (y_true == c)
            if mask.sum() == 0:
                continue
            per_class_recall.append(float((y_pred[mask] == c).sum() / mask.sum()))
        ba = float(np.mean(per_class_recall)) if per_class_recall else 0.0

        # mAP: class-wise AP from probabilities (same style as single_task/main_classify.py)
        aps = []
        for c in range(self.num_classes):
            binary_labels = (y_true == c).astype(np.float32)
            if binary_labels.sum() == 0:
                continue
            scores = y_probs[:, c]
            order = np.argsort(-scores)
            binary_labels = binary_labels[order]
            tp = np.cumsum(binary_labels)
            fp = np.cumsum(1 - binary_labels)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (binary_labels.sum() + 1e-8)
            recall = np.concatenate([[0.0], recall])
            precision = np.concatenate([[1.0], precision])
            ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
            aps.append(float(ap))
        mAP = float(np.mean(aps)) if aps else 0.0
        
        eval_dict = {
            'accuracy': float(accuracy),
            'ba': ba,
            'acc': ba,   # backward compatibility for old logging key
            'mAP': mAP,
            'loss': self.loss / max(self.n_samples, 1),
            'per_class_recall': per_class_recall,
            'per_class_ap': aps,
        }
        
        if verbose:
            print('\nClassification Evaluation')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Balanced Accuracy: {ba:.4f}')
            print(f'mAP: {mAP:.4f}')
            if self.loss > 0:
                print(f'Average Loss: {eval_dict["loss"]:.4f}')
        
        return eval_dict