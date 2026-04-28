# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, ampscaler_get_grad_norm

from mtl_loss_schemes import MultiTaskLoss, get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output
from ptflops import get_model_complexity_info
from models.lora import mark_only_lora_as_trainable

try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb library not found. Logging is disabled.")
    wandb_available = False


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    return obj


def _plot_training_curves(log_entries, output_dir, tasks):
    if plt is None or len(log_entries) == 0:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [e["epoch"] for e in log_entries]
    train_mtl = [e.get("train_mtl_loss") for e in log_entries]
    val_mtl = [e.get("val_mtl_loss") for e in log_entries]
    val_epochs = [ep for ep, v in zip(epochs, val_mtl) if v is not None]
    val_mtl = [v for v in val_mtl if v is not None]

    fig = plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_mtl, label="train_mtl_loss")
    if len(val_mtl) > 0:
        plt.plot(val_epochs, val_mtl, label="val_mtl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MTL Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mtl_loss_curve.png", dpi=150)
    plt.close(fig)

    n_tasks = max(len(tasks), 1)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(8, 3 * n_tasks), squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i, 0]
        train_task = [e.get("train_task_losses", {}).get(task) for e in log_entries]
        train_task = [np.nan if v is None else v for v in train_task]
        val_task = [e.get("val_task_losses", {}).get(task) for e in log_entries]
        val_task_epochs = [ep for ep, v in zip(epochs, val_task) if v is not None]
        val_task = [v for v in val_task if v is not None]
        ax.plot(epochs, train_task, label=f"train/{task}")
        if len(val_task) > 0:
            ax.plot(val_task_epochs, val_task, label=f"val/{task}")
        ax.set_title(f"{task} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task_loss_curves.png", dpi=150)
    plt.close(fig)


def _apply_dinov3_vit_freeze_policy(model):
    # Keep LoRA trainable (already handled by mark_only_lora_as_trainable),
    # freeze deep decoder fusion blocks for ViT to match single_task setup.
    if not hasattr(model, "backbone"):
        return
    bb = model.backbone
    if not hasattr(bb, "backbone_type") or bb.backbone_type != "vit":
        return
    for module in [bb.proj1, bb.proj2, bb.proj3, bb.proj4, bb.fusion1, bb.fusion2]:
        for p in module.parameters():
            p.requires_grad = True
    for module in [bb.fusion3, bb.fusion4]:
        for p in module.parameters():
            p.requires_grad = False


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=10,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='List of tasks to run in MTL setup.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument(
        '--wheat',type=str, help='specify the path to load Wheat, replaces --data-path')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--decoder_map', type=str,
                        help='Path to JSON file containing the type of decoder heads')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Skip loading decoder head weights')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--run_name', type=str,
                        help='wandb run name')
    parser.add_argument('--no_eval_50', action='store_false',
                        help='Disable the iniital eval at 50 epochs.')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    teacher = None
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {n_parameters / 1e6} M")

    model.cuda()
    skip_ptflops = bool(config.MODEL.TYPE == "dinov3" and hasattr(config, "TASKS") and "count" in config.TASKS)
    if skip_ptflops:
        logger.info("Skip ptflops for dinov3+count (PET window partition is input-shape sensitive).")
        macs, params = None, None
    else:
        macs, params = get_model_complexity_info(
            model,
            (3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
    logger.info(f"ptflops GMACS = {macs} and params = {params}")

    model_without_ddp = model

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount(enabled=config.AMP_ENABLE)

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(
            data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
        all_loss_weights={
            'semseg': 20.0,
            'detect':10.0,
            'classify':5.0,
            'count': 5.0,
        }
        loss_weights = {}
        for t in config.TASKS:
            loss_weights[t] = all_loss_weights.get(t, 1.0)

        criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    max_accuracy = 0.0
    training_log_path = os.path.join(config.OUTPUT, "training_log.json")
    training_logs = []

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)

        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model, 0)
        if config.EVAL_MODE:
            return

    if config.MODEL.RESUME_BACKBONE:
        max_accuracy = load_checkpoint(
            config, model_without_ddp.backbone, optimizer, lr_scheduler, loss_scaler, logger, True)
        if config.EVAL_MODE:
            validate(config, data_loader_val, model, 0)
            return

    if config.EVAL_MODE:
        validate(config, data_loader_val, model, 0)
        return

    if dist.get_rank() == 0 and os.path.exists(training_log_path):
        try:
            with open(training_log_path, "r") as f:
                loaded_logs = json.load(f)
            if isinstance(loaded_logs, list):
                training_logs = loaded_logs
        except Exception as e:
            logger.warning(f"Failed to load existing training log: {e}")
            training_logs = []

    if teacher is not None:
        print("loading teacher.......")
        load_checkpoint(config, teacher, optimizer, lr_scheduler,
                        loss_scaler, logger, quiet=True)

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if not config.SKIP_INITIAL_EVAL:
            acc1, _, _ = validate(config, data_loader_val, model, 0)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    if config.MODEL.MTLORA.ENABLED:
        if config.MODEL.MTLORA.FREEZE_PRETRAINED:
            print("\nMarking LoRA params only as trainable:")
            mark_only_lora_as_trainable(model.backbone,
                                        bias=config.MODEL.MTLORA.BIAS,
                                        freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
                                        freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
                                        free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
                                        freeze_downsample_reduction=True if config.MODEL.MTLORA.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION)
            if config.MODEL.TYPE == 'dinov3':
                _apply_dinov3_vit_freeze_policy(model)
        else:
            print("Marking all layers as trainable")
    if config.MODEL.FREEZE_BACKBONE:
        assert (not config.MODEL.MTLORA.ENABLED)
        print("Freezing backbone.........")
        model.freeze_backbone()
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters()
                      if p.requires_grad and 'lora' in name)
    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_params_without_lora = total_model_params - lora_params
    decoder_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'backbone' not in name)

    print(f"""
Number of trainable params: {trainable_params:,}
Decoder params:             {decoder_params:,}
LoRA params:                {lora_params:,}
Extra params:                {(trainable_params - (lora_params + decoder_params)):,}
Total params:               {total_model_params:,} (trainable ratio: {trainable_params/total_model_params * 100:2.2f}%)
Total params without LoRA:  {total_model_params_without_lora:,} (trainable ratio: {trainable_params/total_model_params_without_lora * 100:2.2f}%)
""")
    logger.info("Start training")
    start_time = time.perf_counter()

    start_epoch = int(config.TRAIN.START_EPOCH) if hasattr(config.TRAIN, "START_EPOCH") else 0
    if dist.get_rank() == 0 and len(training_logs) > 0:
        training_logs = [e for e in training_logs if e.get("epoch", -1) < start_epoch]

    epoch = start_epoch
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        if not config.MTL:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, teacher=teacher
        )
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        eval_results = None
        if epoch % config.EVAL_FREQ == 0 or (not args.no_eval_50 and epoch == 50):
            if config.MTL:
                eval_results = validate(config, data_loader_val, model, epoch)
            else:
                acc1, _, _ = validate(config, data_loader_val, model, epoch)
                max_accuracy = max(max_accuracy, acc1)
        if dist.get_rank() == 0:
            epoch_log = {
                "epoch": int(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_mtl_loss": float(train_stats.get("mtl_loss", 0.0)),
                "train_task_losses": _to_serializable(train_stats.get("task_losses", {})),
            }
            if eval_results is not None:
                val_meta = eval_results.pop("_loss_meta", {}) if isinstance(eval_results, dict) else {}
                epoch_log["val_mtl_loss"] = _to_serializable(val_meta.get("val_mtl_loss"))
                epoch_log["val_task_losses"] = _to_serializable(val_meta.get("val_task_losses", {}))
                epoch_log["val_metrics"] = _to_serializable(eval_results)
            training_logs.append(epoch_log)
            with open(training_log_path, "w") as f:
                json.dump(_to_serializable(training_logs), f, indent=2)
            _plot_training_curves(training_logs, config.OUTPUT, config.TASKS if hasattr(config, "TASKS") else [])

    # final eval
    final_eval = validate(config, data_loader_val, model, epoch)
    if dist.get_rank() == 0:
        val_meta = final_eval.pop("_loss_meta", {}) if isinstance(final_eval, dict) else {}
        final_log = {
            "epoch": int(epoch),
            "final_eval": True,
            "val_mtl_loss": _to_serializable(val_meta.get("val_mtl_loss")),
            "val_task_losses": _to_serializable(val_meta.get("val_task_losses", {})),
            "val_metrics": _to_serializable(final_eval),
        }
        training_logs.append(final_log)
        with open(training_log_path, "w") as f:
            json.dump(_to_serializable(training_logs), f, indent=2)
        _plot_training_curves(training_logs, config.OUTPUT, config.TASKS if hasattr(config, "TASKS") else [])
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, task=None, teacher=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    # 建立task_id到task_name的映射 (根据Wheat数据集的task_id定义)
    # task_id 0 -> semseg, task_id 1 -> classify, task_id 2 -> detect, task_id 3 -> count
    task_id_to_name = {0: 'semseg', 1: 'classify', 2: 'detect', 3: 'count'}
    # 反向映射：从batch sample中的键识别任务名称
    sample_key_to_task = {'semseg': 'semseg', 'text': 'classify', 'bbox': 'detect', 'points': 'count'}
    task_loss_meters = {t: AverageMeter() for t in config.TASKS} if config.MTL else {}
    
    # 记录当前accumulation step中已处理的任务，用于清零head梯度
    processed_tasks_in_step = set()

    for idx, batch in enumerate(data_loader):
        if not config.MTL:
            samples, targets = batch
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            # 识别当前batch属于哪个任务
            current_task = None
            if 'sample' in batch:
                # 从sample中识别任务
                sample_keys = batch['sample'].keys()
                for key in sample_keys:
                    if key in sample_key_to_task:
                        current_task = sample_key_to_task[key]
                        break
            elif 'task' in batch:
                # 从task_id识别任务
                task_id = batch['task'][0].item() if isinstance(batch['task'], torch.Tensor) else batch['task'][0]
                current_task = task_id_to_name.get(task_id, None)
            
            # 如果无法识别，尝试从batch的键中识别
            if current_task is None:
                for task_name in config.TASKS:
                    if task_name in batch:
                        current_task = task_name
                        break
            
            if current_task is None:
                logger.warning(f"无法识别batch {idx}的任务，跳过")
                continue

            # 提取当前任务的图像和目标
            if 'sample' in batch:
                samples = batch['sample']['image'].cuda(non_blocking=True)
                # 根据任务类型提取目标
                if current_task == 'semseg':
                    targets = {current_task: batch['sample']['semseg'].cuda(non_blocking=True)}
                elif current_task == 'classify':
                    targets = {current_task: batch['sample']['text'].cuda(non_blocking=True)}
                elif current_task == 'count':
                    # PET 损失需要 list of dict: [{'points': tensor, 'labels': tensor, 'density': scalar}, ...]
                    pts_list = batch['sample']['points']
                    den = batch['sample']['density']
                    if isinstance(den, torch.Tensor):
                        den = den.cuda(non_blocking=True)
                    B = len(pts_list)
                    targets_count = []
                    for i in range(B):
                        p = pts_list[i].cuda(non_blocking=True) if isinstance(pts_list[i], torch.Tensor) else torch.as_tensor(pts_list[i], device=samples.device)
                        n = p.size(0)
                        targets_count.append({
                            'points': p,
                            'labels': torch.ones(n, device=p.device, dtype=torch.long),
                            'density': den[i] if den.dim() > 0 else den,
                        })
                    targets = {current_task: targets_count}
                elif current_task == 'detect':
                    bbox_list = batch['sample']['bbox']
                    batch_size = len(bbox_list)
                    
                    batch_idx_list = []
                    cls_list = []
                    bboxes_list = []
                    
                    for img_idx, bbox in enumerate(bbox_list):
                        # 转换bbox为tensor
                        if isinstance(bbox, np.ndarray):
                            bbox_tensor = torch.from_numpy(bbox)
                        elif isinstance(bbox, torch.Tensor):
                            bbox_tensor = bbox
                        else:
                            bbox_tensor = torch.tensor(bbox)
                        
                        # 确保bbox_tensor是2D的 [N, 5]，格式为 [class, x, y, w, h]
                        if bbox_tensor.numel() == 0:
                            # 空tensor，确保是 [0, 5] 格式
                            bbox_tensor = bbox_tensor.view(0, 5) if bbox_tensor.dim() > 0 else torch.empty(0, 5, dtype=torch.float32)
                        elif bbox_tensor.dim() == 1:
                            # 如果是1D tensor，检查长度
                            if bbox_tensor.size(0) == 5:
                                # 单个bbox，reshape为 [1, 5]
                                bbox_tensor = bbox_tensor.unsqueeze(0)
                            else:
                                # 格式错误，转换为空tensor
                                bbox_tensor = torch.empty(0, 5, dtype=torch.float32)
                        elif bbox_tensor.dim() == 2:
                            # 已经是2D，确保第二维是5
                            if bbox_tensor.size(1) != 5:
                                # 如果第二维不是5，尝试修复
                                if bbox_tensor.size(0) > 0 and bbox_tensor.size(0) == 5:
                                    # 可能是转置了，转回来
                                    bbox_tensor = bbox_tensor.unsqueeze(0)
                                else:
                                    bbox_tensor = torch.empty(0, 5, dtype=torch.float32)
                        
                        num_boxes = bbox_tensor.size(0)
                        if num_boxes > 0:
                            # 提取类别和坐标
                            # bbox_tensor格式: [N, 5] = [class, x, y, w, h]
                            cls = bbox_tensor[:, 0].long()  # [N] 类别
                            bboxes = bbox_tensor[:, 1:5]    # [N, 4] xywh坐标
                            
                            # 为每个bbox添加batch索引
                            batch_idx = torch.full((num_boxes,), img_idx, dtype=torch.long)
                            
                            batch_idx_list.append(batch_idx)
                            cls_list.append(cls)
                            bboxes_list.append(bboxes)
                    
                    # 将所有数据连接成YOLOv8期望的格式
                    if len(batch_idx_list) > 0:
                        batch_idx_tensor = torch.cat(batch_idx_list, dim=0).cuda(non_blocking=True)
                        cls_tensor = torch.cat(cls_list, dim=0).cuda(non_blocking=True)
                        bboxes_tensor = torch.cat(bboxes_list, dim=0).cuda(non_blocking=True)
                    else:
                        # 如果所有图片都没有bbox，创建空tensor
                        batch_idx_tensor = torch.empty(0, dtype=torch.long).cuda(non_blocking=True)
                        cls_tensor = torch.empty(0, dtype=torch.long).cuda(non_blocking=True)
                        bboxes_tensor = torch.empty(0, 4, dtype=torch.float32).cuda(non_blocking=True)
                    
                    # 组织成YOLOv8期望的batch格式
                    targets = {
                        current_task: {
                            "batch_idx": batch_idx_tensor,
                            "cls": cls_tensor,
                            "bboxes": bboxes_tensor
                        }
                    }
            else:
                samples = batch['image'].cuda(non_blocking=True)
                targets = {current_task: batch[current_task].cuda(non_blocking=True)}

        # Mixup通常只用于分类任务，检测任务跳过mixup（仅 MTL 时 current_task 有定义）
        if mixup_fn is not None and (not config.MTL or current_task != 'detect'):
            samples, targets = mixup_fn(samples, targets)
        
        # 前向传播：传入 current_task，backbone 只计算共享 LoRA + 当前任务的 lora_tasks_A/B，并只跑当前任务的 Decoder
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if config.MTL and current_task is not None:
                outputs = model(samples, current_task=current_task)
            else:
                outputs = model(samples)
            
            # 只计算当前任务的loss
            if config.MTL:
                # 只计算当前任务的loss
                task_loss_fn = criterion.loss_ft[current_task]
                task_loss_weight = criterion.loss_weights[current_task]
                if current_task == 'count':
                    task_loss = task_loss_fn(outputs[current_task], targets[current_task], epoch=epoch)
                else:
                    task_loss = task_loss_fn(outputs[current_task], targets[current_task])
                loss = task_loss_weight * task_loss
                loss_dict = {current_task: task_loss}
                if current_task in task_loss_meters:
                    task_loss_meters[current_task].update(task_loss.item() if torch.is_tensor(task_loss) else float(task_loss))
            else:
                loss, loss_dict = criterion(outputs, targets)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        update_grad = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=[p for p in model.parameters() if p.requires_grad],
            create_graph=is_second_order,
            update_grad=update_grad,
        )

        if update_grad:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        if hasattr(loss_scaler, "_scaler") and loss_scaler._scaler is not None:
            scaler_state = loss_scaler.state_dict()
            loss_scale_value = scaler_state.get("scale", 1.0)
        else:
            loss_scale_value = 1.0

        # torch.cuda.synchronize()

        if not config.MTL:
            loss_meter.update(loss.item(), targets.size(0))
        else:
            loss_meter.update(loss.item())

        if grad_norm is not None:  # loss_scaler return None if not update
            # Convert tensor to scalar if needed
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        if wandb_available:
            metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": idx,
                "train/train_loss": loss_meter.val,
                "train/train_loss_avg": loss_meter.avg,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/weight_decay": optimizer.param_groups[0]['weight_decay'],
                "train/time": batch_time.val,
                "train/time_avg": batch_time.avg,
                "train/grad_norm": norm_meter.val,
                "train/grad_norm_avg": norm_meter.avg,
                "train/loss_scale": scaler_meter.val,
                "train/loss_scale_avg": scaler_meter.avg,
                "train/memory": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"train/tasks/{task}/loss"] = task_loss.item()
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            task_info = f"task:{current_task}" if config.MTL and current_task else ""
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t{task_info}\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        # 对于检测任务，需要传递图像尺寸
        processed_output = {}
        for t in config.TASKS:
            if t == 'detect':
                # 从batch中获取图像尺寸（假设所有图像尺寸相同）
                img_size = samples.size(3) if 'samples' in locals() else 448
                processed_output[t] = get_output(outputs[t], t, image_size=img_size)
            else:
                processed_output[t] = get_output(outputs[t], t)
        performance_meter.update(processed_output, targets)

        scores = performance_meter.get_score(verbose=True)
        if wandb_available:
            scores_logs = {
                "train/epoch": epoch,
            }
            if 'semseg' in scores:
                scores_logs["train/tasks/semseg/mIoU"] = scores['semseg']['mIoU']
            if 'classify' in scores:
                scores_logs["train/tasks/classify/ba"] = scores['classify'].get('ba', scores['classify'].get('acc', 0.0))
                scores_logs["train/tasks/classify/mAP"] = scores['classify'].get('mAP', 0.0)
            if 'detect' in scores:
                scores_logs["train/tasks/detect/AP50"] = scores['detect'].get('AP50', 0.0)
                scores_logs["train/tasks/detect/AP75"] = scores['detect'].get('AP75', 0.0)
                scores_logs["train/tasks/detect/AP"] = scores['detect'].get('AP', scores['detect'].get('map', 0.0))
            if 'normals' in loss_dict:
                scores_logs["train/tasks/normals/mean"] = scores['normals']['mean']
                scores_logs["train/tasks/normals/rmse"] = scores['normals']['rmse']
                scores_logs["train/tasks/normals/mean_v2"] = scores['normals']['mean_v2']
                scores_logs["train/tasks/normals/rmse_v2"] = scores['normals']['rmse_v2']
            if 'human_parts' in loss_dict:
                scores_logs["train/tasks/human_parts/mIoU"] = scores['human_parts']['mIoU']
            if 'sal' in loss_dict:
                scores_logs["train/tasks/sal/maxF"] = scores['sal']['maxF']
                scores_logs["train/tasks/sal/Beta maxF"] = scores['sal']['Beta maxF']
                scores_logs["train/tasks/sal/mIoU"] = scores['sal']['mIoU']
            if 'edge' in loss_dict:
                scores_logs["train/tasks/sal/loss"] = scores['edge']['loss']
            if 'depth' in loss_dict:
                scores_logs["train/tasks/depth/rmse"] = scores['depth']['rmse']
                scores_logs["train/tasks/depth/log_rmse"] = scores['depth']['log_rmse']

            wandb.log(scores_logs)

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    if config.MTL:
        return {
            "mtl_loss": float(loss_meter.avg),
            "task_losses": {t: (float(m.avg) if m.count > 0 else None) for t, m in task_loss_meters.items()},
        }
    return {"mtl_loss": float(loss_meter.avg), "task_losses": {}}


@torch.no_grad()
def validate(config, data_loader, model, epoch):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = config.TASKS
    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)
    loss_meter = AverageMeter()

    # 为每个任务构建 loss 函数和权重
    loss_ft = torch.nn.ModuleDict(
        {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
    all_loss_weights = {
        'semseg': 20.0,
        'detect': 10.0,
        'classify': 5.0,
        'count': 5.0,
    }
    loss_weights = {t: all_loss_weights.get(t, 1.0) for t in config.TASKS}
    task_loss_meters = {t: AverageMeter() for t in config.TASKS}

    # MTL 验证时，Wheat 数据集的 batch 结构与训练一致：每个 batch 只包含一种任务
    # 这里复用 train_one_epoch 中的任务识别逻辑
    task_id_to_name = {0: 'semseg', 1: 'classify', 2: 'detect', 3: 'count'}
    sample_key_to_task = {'semseg': 'semseg', 'text': 'classify', 'bbox': 'detect', 'points': 'count'}

    model.eval()
    num_val_points = 0
    logger.info("Start eval")
    start = time.perf_counter()
    loss_dict = None
    # 统计各任务在验证中实际出现的 batch 数，便于排查某任务指标为 0 的情况
    seen_task_batches = {t: 0 for t in config.TASKS}
    skipped_unrecognized = 0
    
    # 计算总batch数用于进度显示
    total_batches = len(data_loader)
    logger.info(f"Total validation batches: {total_batches}")

    for i, batch in enumerate(data_loader):
        # 每10个batch打印一次进度
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            elapsed = time.perf_counter() - start
            logger.info(f"Eval progress: {i+1}/{total_batches} batches, elapsed: {elapsed:.2f}s")
        # 某些数据集/自定义 DataLoader 可能没有 'meta' 字段，这里做安全检查（主要兼容 NYUD/PASCAL）
        try:
            if isinstance(batch, dict) and 'meta' in batch and 'image' in batch['meta']:
                logger.debug(f"Image ID = {batch['meta']['image']}")
        except Exception:
            pass

        # 针对 Wheat + MTL 的 batch 结构（与训练相同）
        if config.MTL and isinstance(batch, dict) and 'sample' in batch:
            current_task = None
            # 1) 优先从 sample 内的 key 推断任务
            sample_keys = batch['sample'].keys()
            for key in sample_keys:
                if key in sample_key_to_task:
                    current_task = sample_key_to_task[key]
                    break

            # 2) 如果还没识别到，再从 task / task_id 推断
            if current_task is None:
                if 'task' in batch:
                    task_id = batch['task'][0].item() if isinstance(batch['task'], torch.Tensor) else batch['task'][0]
                    current_task = task_id_to_name.get(task_id, None)
                elif 'task_id' in batch:
                    task_id = batch['task_id'][0].item() if isinstance(batch['task_id'], torch.Tensor) else batch['task_id'][0]
                    current_task = task_id_to_name.get(task_id, None)

            if current_task is None:
                # 最后尝试从 batch 顶层 key 中识别
                for task_name in config.TASKS:
                    if task_name in batch:
                        current_task = task_name
                        break

            if current_task is None:
                logger.warning(f"[VAL] 无法识别 batch {i} 的任务，跳过该 batch")
                skipped_unrecognized += 1
                continue
            if current_task in seen_task_batches:
                seen_task_batches[current_task] += 1

            # 取出图像和对应任务的标签
            samples = batch['sample']['image'].cuda(non_blocking=True)
            if current_task == 'semseg':
                targets = {current_task: batch['sample']['semseg'].cuda(non_blocking=True)}
                targets_for_eval = targets  # 非检测任务，使用相同格式
            elif current_task == 'classify':
                targets = {current_task: batch['sample']['text'].cuda(non_blocking=True)}
                targets_for_eval = targets
            elif current_task == 'count':
                pts_list = batch['sample']['points']
                den = batch['sample']['density']
                if isinstance(den, torch.Tensor):
                    den = den.cuda(non_blocking=True)
                B = len(pts_list)
                targets_count = []
                for i in range(B):
                    p = pts_list[i].cuda(non_blocking=True) if isinstance(pts_list[i], torch.Tensor) else torch.as_tensor(pts_list[i], device=samples.device)
                    targets_count.append({'points': p, 'labels': torch.ones(p.size(0), device=p.device, dtype=torch.long), 'density': den[i] if den.dim() > 0 else den})
                targets = {current_task: targets_count}
                targets_for_eval = {current_task: pts_list}
            elif current_task == 'detect':
                bbox_list = batch['sample']['bbox']
                batch_size = len(bbox_list)
                
                batch_idx_list = []
                cls_list = []
                bboxes_list = []
                bbox_tensors_for_eval = []  # 用于评估的列表格式
                
                for img_idx, bbox in enumerate(bbox_list):
                    # 转换bbox为tensor
                    if isinstance(bbox, np.ndarray):
                        bbox_tensor = torch.from_numpy(bbox)
                    elif isinstance(bbox, torch.Tensor):
                        bbox_tensor = bbox
                    else:
                        bbox_tensor = torch.tensor(bbox)
                    
                    # 确保bbox_tensor是2D的 [N, 5]，格式为 [class, x, y, w, h]
                    if bbox_tensor.numel() == 0:
                        bbox_tensor = bbox_tensor.view(0, 5) if bbox_tensor.dim() > 0 else torch.empty(0, 5, dtype=torch.float32)
                    elif bbox_tensor.dim() == 1:
                        if bbox_tensor.size(0) == 5:
                            bbox_tensor = bbox_tensor.unsqueeze(0)
                        else:
                            bbox_tensor = torch.empty(0, 5, dtype=torch.float32)
                    elif bbox_tensor.dim() == 2:
                        if bbox_tensor.size(1) != 5:
                            if bbox_tensor.size(0) > 0 and bbox_tensor.size(0) == 5:
                                bbox_tensor = bbox_tensor.unsqueeze(0)
                            else:
                                bbox_tensor = torch.empty(0, 5, dtype=torch.float32)
                    
                    # 保存用于评估的格式（列表，每个图像一个元素）
                    bbox_tensors_for_eval.append(bbox_tensor.cuda(non_blocking=True))
                    
                    num_boxes = bbox_tensor.size(0)
                    if num_boxes > 0:
                        # 提取类别和坐标
                        cls = bbox_tensor[:, 0].long()  # [N] 类别
                        bboxes = bbox_tensor[:, 1:5]     # [N, 4] xywh坐标
                        
                        # 为每个bbox添加batch索引
                        batch_idx = torch.full((num_boxes,), img_idx, dtype=torch.long)
                        
                        batch_idx_list.append(batch_idx)
                        cls_list.append(cls)
                        bboxes_list.append(bboxes)
                
                # 将所有数据连接成YOLOv8期望的格式
                if len(batch_idx_list) > 0:
                    batch_idx_tensor = torch.cat(batch_idx_list, dim=0).cuda(non_blocking=True)
                    cls_tensor = torch.cat(cls_list, dim=0).cuda(non_blocking=True)
                    bboxes_tensor = torch.cat(bboxes_list, dim=0).cuda(non_blocking=True)
                else:
                    # 如果所有图片都没有bbox，创建空tensor
                    batch_idx_tensor = torch.empty(0, dtype=torch.long).cuda(non_blocking=True)
                    cls_tensor = torch.empty(0, dtype=torch.long).cuda(non_blocking=True)
                    bboxes_tensor = torch.empty(0, 4, dtype=torch.float32).cuda(non_blocking=True)
                
                # 组织成YOLOv8期望的batch格式（用于loss计算）
                targets = {
                    current_task: {
                        "batch_idx": batch_idx_tensor,
                        "cls": cls_tensor,
                        "bboxes": bboxes_tensor
                    }
                }
                # 列表格式用于评估
                targets_for_eval = {current_task: bbox_tensors_for_eval}
            else:
                logger.warning(f"[VAL] 未知任务 {current_task}，跳过该 batch")
                continue

            # 前向 & 单任务 loss：传入 current_task，只计算共享 LoRA + 当前任务的 lora_tasks_A/B
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples, current_task=current_task)
                task_loss_fn = loss_ft[current_task]
                task_loss_weight = loss_weights[current_task]
                if current_task == 'count':
                    # 验证时 model.eval()，PETCountHead 返回推理格式（无 sparse/dense），无法算 CountLoss，仅用 0 占位
                    task_loss = 0.0
                else:
                    task_loss = task_loss_fn(outputs[current_task], targets[current_task])
                loss = task_loss_weight * (task_loss.item() if torch.is_tensor(task_loss) else task_loss)
                loss_meter.update(loss)
                loss_dict = {current_task: task_loss}
                task_loss_meters[current_task].update(task_loss.item() if torch.is_tensor(task_loss) else float(task_loss))

            # 性能指标：只对当前任务更新
            # 对于检测任务，传递图像尺寸用于坐标转换
            if current_task == 'detect':
                processed_output = {current_task: get_output(outputs[current_task], current_task, image_size=samples.size(3))}
            else:
                processed_output = {current_task: get_output(outputs[current_task], current_task)}
            # 直接更新当前任务的 meter，而不是通过 PerformanceMeter.update()（它期望所有任务）
            # 对于检测任务，使用列表格式的 GT；对于其他任务，使用原始格式
            if current_task == 'detect':
                performance_meter.meters[current_task].update(
                    processed_output[current_task], targets_for_eval[current_task],
                    image_size=(samples.size(2), samples.size(3))
                )
            elif current_task == 'count':
                performance_meter.meters[current_task].update(
                    processed_output[current_task], targets_for_eval[current_task]
                )
            else:
                performance_meter.meters[current_task].update(
                    processed_output[current_task], targets[current_task]
                )

        else:
            # 兼容原始 NYUD/PASCAL 的多任务 batch 结构：batch['image'] + 各任务标签
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

            outputs = model(images)
            num_val_points += 1

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                # 对所有任务一起计算 MultiTaskLoss
                total_loss = 0.0
                loss_dict = {}
                for t in tasks:
                    tl = loss_ft[t](outputs[t], targets[t])
                    loss_dict[t] = tl
                    total_loss += loss_weights[t] * tl
                loss = total_loss
                loss_meter.update(loss.item())

            # 对于检测任务，需要传递图像尺寸
            processed_output = {}
            for t in tasks:
                if t == 'detect':
                    img_size = images.size(3)  # 从图像张量获取尺寸
                    processed_output[t] = get_output(outputs[t], t, image_size=img_size)
                else:
                    processed_output[t] = get_output(outputs[t], t)
            performance_meter.update(processed_output, targets)

        # wandb 日志
        if wandb_available:
            metrics = {
                "val/epoch_ndx": epoch,
                "val/batch_ndx": i,
                "val/val_loss": loss_meter.val,
                "val/val_loss_avg": loss_meter.avg,
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"val/tasks/{task}/loss"] = task_loss.item()
            wandb.log(metrics)

    logger.info(f"val loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t")

    eval_results = performance_meter.get_score(verbose=True)
    # 将验证指标写入 log（log_rank0.txt），便于后续检索
    try:
        logger.info(f"[VAL] seen_task_batches={seen_task_batches}, skipped_unrecognized={skipped_unrecognized}")
        for t in config.TASKS:
            if t in eval_results:
                logger.info(f"[VAL] {t}: {eval_results[t]}")
    except Exception as e:
        logger.warning(f"[VAL] Failed to log eval_results: {e}")
    epoch_time = time.perf_counter() - start
    logger.info(
        f"eval takes {datetime.timedelta(seconds=int(epoch_time))}")
    if wandb_available:
        scores_logs = {
            "val/epoch": epoch,
        }
        if 'semseg' in eval_results:
            scores_logs["val/tasks/semseg/mIoU"] = eval_results['semseg']['mIoU']
        if 'classify' in eval_results:
            scores_logs["val/tasks/classify/ba"] = eval_results['classify'].get('ba', eval_results['classify'].get('acc', 0.0))
            scores_logs["val/tasks/classify/mAP"] = eval_results['classify'].get('mAP', 0.0)
        if 'detect' in eval_results:
            scores_logs["val/tasks/detect/AP50"] = eval_results['detect'].get('AP50', 0.0)
            scores_logs["val/tasks/detect/AP75"] = eval_results['detect'].get('AP75', 0.0)
            scores_logs["val/tasks/detect/AP"] = eval_results['detect'].get('AP', eval_results['detect'].get('map', 0.0))
        if 'normals' in eval_results:
            scores_logs["val/tasks/normals/mean"] = eval_results['normals']['mean']
            scores_logs["val/tasks/normals/rmse"] = eval_results['normals']['rmse']
            scores_logs["val/tasks/normals/mean_v2"] = eval_results['normals']['mean_v2']
            scores_logs["val/tasks/normals/rmse_v2"] = eval_results['normals']['rmse_v2']
        if 'human_parts' in eval_results:
            scores_logs["val/tasks/human_parts/mIoU"] = eval_results['human_parts']['mIoU']
        if 'sal' in eval_results:
            scores_logs["val/tasks/sal/maxF"] = eval_results['sal']['maxF']
            scores_logs["val/tasks/sal/Beta maxF"] = eval_results['sal']['Beta maxF']
            scores_logs["val/tasks/sal/mIoU"] = eval_results['sal']['mIoU']
        if 'edge' in eval_results:
            scores_logs["val/tasks/sal/loss"] = eval_results['edge']['loss']
        if 'depth' in eval_results:
            scores_logs["val/tasks/depth/rmse"] = eval_results['depth']['rmse']
            scores_logs["val/tasks/depth/log_rmse"] = eval_results['depth']['log_rmse']

        wandb.log(scores_logs)

    eval_results["_loss_meta"] = {
        "val_mtl_loss": float(loss_meter.avg),
        "val_task_losses": {t: (float(m.avg) if m.count > 0 else None) for t, m in task_loss_meters.items()},
    }
    return eval_results


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        # torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.perf_counter()
        for i in range(30):
            model(images)
        # torch.cuda.synchronize()
        tic2 = time.perf_counter()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.disable_wandb:
        wandb_available = False
        logger.info("Wandb logging disabled.")
    elif wandb_available:
        try:
            if not os.getenv("WANDB_API_KEY"):
                wandb.login()
            else:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
            config_name = f"{os.path.basename(os.path.dirname(args.cfg))}/{os.path.basename(args.cfg)}"
            wandb.init(project='MTLoRA', config=config,
                       name=config_name if not args.run_name else args.run_name)
            wandb.config.update({'args': vars(args)})
        except wandb.exc.LaunchError:
            logger.warnning("Could not initialize wandb. Logging is disabled.")
            wandb_available = False

    main(config)
