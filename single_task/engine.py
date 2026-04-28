"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import pet_head.misc as utils
from pet_head.misc import NestedTensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, targets, pred, vis_dir, split_map=None):
    """
    Visualize predictions
    """
    gts = [t['points'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw ground-truth points (red)
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1)

        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)


# training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets]

        outputs = model(samples, epoch=epoch, train=True, 
                                        criterion=criterion, targets=targets)
        loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # For R² calculation
    all_gt_counts = []
    all_pred_counts = []

    print_freq = 10
    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]

        # inference
        outputs = model(samples, test=True, targets=targets)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        outputs_offsets = outputs['pred_offsets'][0]

        # DEBUG: Print prediction info for first sample
        if idx == 0:
            print(f"[EVAL DEBUG] Total predictions: {len(outputs_scores)}")
            if len(outputs_scores) > 0:
                print(f"[EVAL DEBUG] Score range: [{outputs_scores.min():.4f}, {outputs_scores.max():.4f}]")
                print(f"[EVAL DEBUG] Scores > 0.5: {(outputs_scores > 0.5).sum().item()}")
            else:
                print(f"[EVAL DEBUG] WARNING: No predictions output!")

        # process predicted points (apply threshold filtering)
        thrs = 0.5
        valid_idx = outputs_scores > thrs
        predict_cnt = valid_idx.sum().item()
        gt_cnt = targets[0]['points'].shape[0]

        # Store for R² calculation
        all_pred_counts.append(predict_cnt)
        all_gt_counts.append(gt_cnt)

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

        # record results
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda()
        results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        metric_logger.update(mae=results['mae'], mse=results['mse'])

        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'])

        # visualize predictions
        if vis_dir: 
            points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
            split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir, split_map=split_map)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Calculate RMSE
    results['rmse'] = np.sqrt(results['mse'])

    # Calculate R²
    all_gt_counts = np.array(all_gt_counts)
    all_pred_counts = np.array(all_pred_counts)

    gt_mean = np.mean(all_gt_counts)
    ss_tot = np.sum((all_gt_counts - gt_mean) ** 2)
    ss_res = np.sum((all_gt_counts - all_pred_counts) ** 2)

    if ss_tot > 0:
        results['r2'] = 1 - (ss_res / ss_tot)
    else:
        results['r2'] = 0.0

    return results