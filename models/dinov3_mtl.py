import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_blocks import FeatureFusionBlock_custom
from .fcos_head import FCOSHead, compute_locations
from .lora import MTLoRALinear
from .pet_head import PETCountHead


_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_single_task_root = os.path.join(_repo_root, "single_task")
if _single_task_root not in sys.path:
    sys.path.insert(0, _single_task_root)

from dinov3.backbone import DINOv3ContextBackbone, DINOv3ViTBackbone


class MTLoRAAdapter(nn.Module):
    """
    Wrap MTLoRALinear to expose a plain forward(x) API.
    - ta mode: use shared output
    - ts mode: use task-specific output when current_task is set
    """
    def __init__(self, linear: MTLoRALinear, mode: str = "ta"):
        super().__init__()
        assert mode in {"ta", "ts"}
        self.linear = linear
        self.mode = mode
        self.current_task = None
        self.in_features = linear.linear.in_features
        self.out_features = linear.linear.out_features

    def set_current_task(self, current_task: str):
        self.current_task = current_task

    def forward(self, x: torch.Tensor):
        shared_out, task_out = self.linear(x, current_task=self.current_task)
        if self.mode == "ts" and task_out is not None and self.current_task is not None and self.current_task in task_out:
            return task_out[self.current_task]
        return shared_out


def _build_stage_block_groups(end_indices: List[int], total_blocks: int) -> List[List[int]]:
    groups = []
    start = 0
    for end in end_indices:
        if end < start or end >= total_blocks:
            continue
        groups.append(list(range(start, end + 1)))
        start = end + 1
    if start < total_blocks:
        groups.append(list(range(start, total_blocks)))
    return groups


class DPTSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x: torch.Tensor):
        return self.head(x)


class GAPMLPHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)


class DINOv3DPTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        dcfg = config.MODEL.DINOV3
        model_name = dcfg.MODEL_NAME
        pretrained_path = dcfg.PRETRAINED
        self.tasks = list(config.TASKS) if hasattr(config, "TASKS") else []
        self._mtlora_adapters: List[MTLoRAAdapter] = []
        self._cfg = config
        if "vit" in model_name:
            self.backbone = DINOv3ViTBackbone(model_name=model_name, pretrained_path=pretrained_path, patch_size=dcfg.PATCH_SIZE)
            self.backbone_type = "vit"
            vit_features = self.backbone.embed_dim
            backbone_channels = [96, 192, 384, 768]
            self.reassemble1 = nn.Sequential(
                nn.Conv2d(vit_features, backbone_channels[0], kernel_size=1),
                nn.ConvTranspose2d(backbone_channels[0], backbone_channels[0], kernel_size=4, stride=4),
            )
            self.reassemble2 = nn.Sequential(
                nn.Conv2d(vit_features, backbone_channels[1], kernel_size=1),
                nn.ConvTranspose2d(backbone_channels[1], backbone_channels[1], kernel_size=2, stride=2),
            )
            self.reassemble3 = nn.Sequential(nn.Conv2d(vit_features, backbone_channels[2], kernel_size=1))
            self.reassemble4 = nn.Sequential(
                nn.Conv2d(vit_features, backbone_channels[3], kernel_size=1),
                nn.Conv2d(backbone_channels[3], backbone_channels[3], kernel_size=3, stride=2, padding=1),
            )
            if config.MODEL.MTLORA.ENABLED:
                self._inject_mtlora_vit(config)
        else:
            self.backbone = DINOv3ContextBackbone(model_name=model_name, pretrained_path=pretrained_path)
            self.backbone_type = "convnext"
            backbone_channels = self.backbone.out_channels
            if config.MODEL.MTLORA.ENABLED:
                self._inject_mtlora_convnext(config)

        hidden_dim = dcfg.HIDDEN_DIM
        self.proj1 = nn.Conv2d(backbone_channels[0], hidden_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(backbone_channels[1], hidden_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(backbone_channels[2], hidden_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(backbone_channels[3], hidden_dim, kernel_size=1)

        self.fusion1 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))
        self.fusion2 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))
        self.fusion3 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))
        self.fusion4 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))

    def _new_mtlora_linear(self, old_linear: nn.Linear, stage_id: int, mode: str):
        mtl_cfg = self._cfg.MODEL.MTLORA
        dropout = mtl_cfg.DROPOUT[stage_id] if isinstance(mtl_cfg.DROPOUT, list) else float(mtl_cfg.DROPOUT)
        shared_scale = mtl_cfg.SHARED_SCALE[stage_id] if isinstance(mtl_cfg.SHARED_SCALE, list) else float(mtl_cfg.SHARED_SCALE)
        if hasattr(mtl_cfg, "R_PER_TASK_LIST") and len(mtl_cfg.R_PER_TASK_LIST) > 0:
            r_map = dict(mtl_cfg.R_PER_TASK_LIST[stage_id])
        else:
            r_shared = mtl_cfg.R[stage_id] if isinstance(mtl_cfg.R, list) else int(mtl_cfg.R)
            r_map = {"shared": r_shared}
            for t in self.tasks:
                r_map[t] = r_shared
        if hasattr(mtl_cfg, "SCALE_PER_TASK_LIST") and len(mtl_cfg.SCALE_PER_TASK_LIST) > 0:
            task_scale_map = dict(mtl_cfg.SCALE_PER_TASK_LIST[stage_id])
        else:
            task_scale = mtl_cfg.TASK_SCALE[stage_id] if isinstance(mtl_cfg.TASK_SCALE, list) else float(mtl_cfg.TASK_SCALE)
            task_scale_map = {t: task_scale for t in self.tasks}

        tasks = None if mode == "ta" else self.tasks
        r = {"shared": r_map["shared"]} if mode == "ta" else r_map
        lora = MTLoRALinear(
            old_linear.in_features,
            old_linear.out_features,
            r=r,
            lora_shared_scale=shared_scale,
            lora_task_scale=task_scale_map,
            lora_dropout=dropout,
            tasks=tasks,
            trainable_scale_shared=mtl_cfg.TRAINABLE_SCALE_SHARED,
            trainable_scale_per_task=mtl_cfg.TRAINABLE_SCALE_PER_TASK,
            shared_mode=mtl_cfg.SHARED_MODE,
            bias=(old_linear.bias is not None),
        )
        lora.linear.weight.data.copy_(old_linear.weight.data)
        if old_linear.bias is not None:
            lora.linear.bias.data.copy_(old_linear.bias.data)
        adapter = MTLoRAAdapter(lora, mode=mode)
        self._mtlora_adapters.append(adapter)
        return adapter

    def _inject_mtlora_vit(self, config):
        blocks = self.backbone.model.blocks
        stage_groups = _build_stage_block_groups(list(self.backbone.out_indices), len(blocks))
        for stage_id, block_indices in enumerate(stage_groups):
            last_idx = block_indices[-1]
            for blk_idx in block_indices:
                block = blocks[blk_idx]
                is_last = blk_idx == last_idx

                # Attention qkv: always TA-LoRA
                if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
                    block.attn.qkv = self._new_mtlora_linear(block.attn.qkv, stage_id, mode="ta")
                # Attention proj: TA for first n-1, TS for the last block
                if hasattr(block.attn, "proj") and isinstance(block.attn.proj, nn.Linear):
                    block.attn.proj = self._new_mtlora_linear(block.attn.proj, stage_id, mode="ts" if is_last else "ta")
                # MLP fc1/fc2: TA for first n-1, TS for the last block
                if hasattr(block.mlp, "fc1") and isinstance(block.mlp.fc1, nn.Linear):
                    block.mlp.fc1 = self._new_mtlora_linear(block.mlp.fc1, stage_id, mode="ts" if is_last else "ta")
                if hasattr(block.mlp, "fc2") and isinstance(block.mlp.fc2, nn.Linear):
                    block.mlp.fc2 = self._new_mtlora_linear(block.mlp.fc2, stage_id, mode="ts" if is_last else "ta")

    def _inject_mtlora_convnext(self, config):
        del config
        # ConvNeXt: each stage has several residual blocks; inject on pwconv1/pwconv2.
        # Stage rule: first n-1 blocks -> TA-LoRA, last block -> TS-LoRA.
        for stage_id, stage in enumerate(self.backbone.model.stages):
            if len(stage) == 0:
                continue
            last_idx = len(stage) - 1
            for blk_idx, block in enumerate(stage):
                is_last = blk_idx == last_idx
                mode = "ts" if is_last else "ta"
                if hasattr(block, "pwconv1") and isinstance(block.pwconv1, nn.Linear):
                    block.pwconv1 = self._new_mtlora_linear(block.pwconv1, stage_id, mode=mode)
                if hasattr(block, "pwconv2") and isinstance(block.pwconv2, nn.Linear):
                    block.pwconv2 = self._new_mtlora_linear(block.pwconv2, stage_id, mode=mode)

    @staticmethod
    def _reshape_vit_tokens(tokens: torch.Tensor, h: int, w: int):
        bsz, _, c = tokens.shape
        expected_n = h * w
        tokens = tokens[:, -expected_n:]
        tokens = tokens.transpose(1, 2)
        return tokens.reshape(bsz, c, h, w)

    def forward(self, x: torch.Tensor, return_stages: bool = True, current_task: str = None):
        for adapter in self._mtlora_adapters:
            adapter.set_current_task(current_task)
        if self.backbone_type == "vit":
            bsz, _, h, w = x.shape
            patch_size = self.backbone.patch_size
            n_h, n_w = h // patch_size, w // patch_size
            layers = self.backbone.model.get_intermediate_layers(x, n=self.backbone.out_indices, reshape=False)
            feat_1 = self.reassemble1(self._reshape_vit_tokens(layers[0], n_h, n_w))
            feat_2 = self.reassemble2(self._reshape_vit_tokens(layers[1], n_h, n_w))
            feat_3 = self.reassemble3(self._reshape_vit_tokens(layers[2], n_h, n_w))
            feat_4 = self.reassemble4(self._reshape_vit_tokens(layers[3], n_h, n_w))
        else:
            feat_1, feat_2, feat_3, feat_4 = self.backbone.model.get_intermediate_layers(x, n=4, reshape=True)

        feat_1 = self.proj1(feat_1)
        feat_2 = self.proj2(feat_2)
        feat_3 = self.proj3(feat_3)
        feat_4 = self.proj4(feat_4)

        feat_4 = self.fusion4(feat_4)
        feat_3 = self.fusion3(feat_4, feat_3)
        feat_2 = self.fusion2(feat_3, feat_2)
        feat_1 = self.fusion1(feat_2, feat_1)

        feats = [feat_1, feat_2, feat_3, feat_4]
        if return_stages:
            return feats
        return feat_1


class MultiTaskDINO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tasks = config.TASKS
        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.fpn_strides = list(config.MODEL.DINOV3.FPN_STRIDES)
        self.backbone = DINOv3DPTBackbone(config)
        hidden_dim = config.MODEL.DINOV3.HIDDEN_DIM

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            if task == "classify":
                self.heads[task] = GAPMLPHead(hidden_dim, self.num_outputs[task])
            elif task == "detect":
                self.heads[task] = FCOSHead(in_channels=hidden_dim, num_classes=self.num_outputs[task], num_levels=len(self.fpn_strides))
            elif task == "count":
                self.heads[task] = PETCountHead(
                    embed_dim=hidden_dim,
                    stage0_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_classes=1,
                    img_size=config.DATA.IMG_SIZE,
                    pet_res_sparse=config.MODEL.PET_COUNT_RES_SPARSE,
                    pet_res_dense=config.MODEL.PET_COUNT_RES_DENSE,
                    use_single_task_recipe=True,
                )
            else:
                self.heads[task] = DPTSegHead(hidden_dim, self.num_outputs[task])

    def _forward_task(self, task: str, feats: List[torch.Tensor], image: torch.Tensor):
        if task == "classify":
            return self.heads[task](feats[0])
        if task == "detect":
            fpn_feats = [feats[1], feats[2], feats[3], F.max_pool2d(feats[3], kernel_size=2, stride=2)]
            logits, bbox_reg, centerness = self.heads[task](fpn_feats)
            locations = compute_locations(fpn_feats, self.fpn_strides)
            return {
                "logits": logits,
                "bbox_reg": bbox_reg,
                "centerness": centerness,
                "locations": locations,
                "image_size": (image.shape[2], image.shape[3]),
            }
        if task == "count":
            return self.heads[task](feats, image, train=self.training)
        return self.heads[task](feats[0])

    def forward(self, x: torch.Tensor, current_task: str = None):
        feats = self.backbone(x, return_stages=True, current_task=current_task)
        if current_task is not None and current_task in self.tasks:
            return {current_task: self._forward_task(current_task, feats, x)}
        out = {}
        for task in self.tasks:
            out[task] = self._forward_task(task, feats, x)
        return out
