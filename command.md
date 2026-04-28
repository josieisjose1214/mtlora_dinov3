# 常用命令

## 1. DINOv3 ConvNeXt 四任务训练

```bash
torchrun --nproc_per_node=1 main.py ^
  --cfg configs/mtlora/tiny_448/mtlora_plus_tiny_448_r16_scale4_pertask.yaml ^
  --wheat /path/to/wheat_root ^
  --tasks semseg,classify,detect,count ^
  --epochs 300 ^
  --eval-freq 5 ^
  --output output ^
  --tag dinov3_convnext_mtl ^
  --opts ^
  MODEL.TYPE dinov3 ^
  MODEL.NAME dinov3_convnext_mtlora ^
  MODEL.DINOV3.MODEL_NAME convnext_small ^
  MODEL.DINOV3.PRETRAINED /path/to/dinov3_convnext_small_pretrain.pth ^
  MODEL.DECODER_HEAD.detect fcos ^
  MODEL.MTLORA.ENABLED True ^
  MODEL.MTLORA.FREEZE_PRETRAINED True
```

## 2. DINOv3 ViT 四任务训练

```bash
torchrun --nproc_per_node=1 main.py ^
  --cfg configs/mtlora/tiny_448/mtlora_plus_tiny_448_r16_scale4_pertask.yaml ^
  --wheat /path/to/wheat_root ^
  --tasks semseg,classify,detect,count ^
  --epochs 120 ^
  --eval-freq 2 ^
  --output output ^
  --tag dinov3_vit_mtl ^
  --opts ^
  MODEL.TYPE dinov3 ^
  MODEL.NAME dinov3_vit_mtlora ^
  MODEL.DINOV3.MODEL_NAME vit_small ^
  MODEL.DINOV3.PRETRAINED /path/to/dinov3_vits16_pretrain.pth ^
  MODEL.DECODER_HEAD.detect fcos ^
  MODEL.MTLORA.ENABLED True ^
  MODEL.MTLORA.FREEZE_PRETRAINED True
```

## 3. 从 checkpoint 继续训练

```bash
torchrun --nproc_per_node=1 main.py ^
  --cfg configs/mtlora/tiny_448/mtlora_plus_tiny_448_r16_scale4_pertask.yaml ^
  --wheat /path/to/wheat_root ^
  --tasks semseg,classify,detect,count ^
  --resume /path/to/checkpoint.pth ^
  --output output ^
  --tag resume_run ^
  --opts ^
  MODEL.TYPE dinov3 ^
  MODEL.DINOV3.MODEL_NAME vit_small ^
  MODEL.DECODER_HEAD.detect fcos ^
  MODEL.MTLORA.ENABLED True
```

## 4. 仅验证

```bash
torchrun --nproc_per_node=1 main.py ^
  --cfg configs/mtlora/tiny_448/mtlora_plus_tiny_448_r16_scale4_pertask.yaml ^
  --wheat /path/to/wheat_root ^
  --tasks semseg,classify,detect,count ^
  --eval ^
  --resume /path/to/checkpoint.pth ^
  --opts ^
  MODEL.TYPE dinov3 ^
  MODEL.DINOV3.MODEL_NAME convnext_small ^
  MODEL.DECODER_HEAD.detect fcos ^
  MODEL.MTLORA.ENABLED True
```

## 5. 输出文件

- `training_log.json`：epoch 级别训练和验证记录
- `mtl_loss_curve.png`：总损失曲线
- `task_loss_curves.png`：各任务损失曲线
- `ckpt_epoch_*.pth`：checkpoint
