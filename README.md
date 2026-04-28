# DINOv3-MTLoRA

基于 MTLoRA 改造的多任务学习项目。当前主线实现将原始 MTLoRA 的 backbone 替换为 DINOv3，并在 backbone 后接 DPT decoder，再接四个任务头，支持以下 Wheat 场景任务：

- `semseg`：语义分割
- `classify`：病害分类
- `detect`：目标检测
- `count`：目标计数

当前仓库的主目标不是复现原始 MTLoRA 论文，而是提供一个可训练、可扩展的 DINOv3 多任务框架，支持：

- DINOv3 ConvNeXt backbone
- DINOv3 ViT backbone
- MTLoRA 注入到 DINOv3 backbone
- DPT 风格多尺度特征融合
- 四任务共享 backbone、按任务独立 head

## 1. 项目结构

```text
.
├─ main.py                    # 训练 / 验证入口
├─ config.py                  # 配置系统
├─ models/
│  ├─ dinov3_mtl.py           # DINOv3 + DPT + 四任务头主实现
│  ├─ lora.py                 # MTLoRA 核心模块
│  ├─ fcos_head.py            # 检测头
│  ├─ pet_head.py             # 计数头
│  └─ dpt_blocks.py           # DPT 融合模块
├─ data/
│  ├─ mtl_ds.py               # 多任务数据构建
│  └─ batcher.py              # Wheat 四任务数据集与 batch sampler
├─ evaluation/                # 各任务评估逻辑
├─ configs/mtlora/tiny_448/   # 训练配置
├─ single_task/               # DINOv3 backbone 及部分单任务复用代码
│  └─ fcos_core/              # 需将FCOS官方仓库中的fcos_core放置此处
└─ command.md                 # 常用训练 / 验证命令
```
**注意**：需从https://github.com/tianzhi0549/FCOS/tree/master/fcos_core 将代码下载到single_task下

## 2. 环境依赖

建议环境：

- Python 3.10
- PyTorch 2.x
- CUDA 11.8 或更高

安装依赖：

```bash
pip install -r requirements.txt
```

说明：

- `single_task/dinov3/` 中包含当前项目依赖的 DINOv3 backbone 实现，因此本仓库可以独立运行，不再依赖外部 `single_task` 仓库。
- 若不使用 Swin 相关旧路径，可不编译 `kernels/window_process/`。

## 3. 数据组织

`--wheat` 指向一个总数据目录，该目录下默认包含四个子任务数据集。

```text
your_wheat_root/
├─ segmentation_dataset/
│  ├─ train/
│  │  ├─ images/
│  │  └─ class_id/
│  └─ val/
│     ├─ images/
│     └─ class_id/
├─ class_disease/
│  └─ classification_dataset/
│     ├─ train/
│     │  ├─ BrownRust/
│     │  ├─ Healthy/
│     │  └─ ...
│     └─ val/
├─ detect_dataset/
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
│     ├─ train/
│     └─ val/
└─ count_dataset/
   ├─ images/
   │  ├─ train/
   │  └─ val/
   └─ annotations/
      ├─ train/
      └─ val/
```

其中：

- 分割标签来自 `segmentation_dataset/*/class_id/`
- 分类数据按 `ImageFolder` 方式组织
- 检测标签采用 YOLO 格式：`class x y w h`
- 计数标签采用 XML，每个目标由框中心转换为点监督

当前 Wheat 任务默认类别数：

- `semseg`: 3
- `classify`: 8
- `detect`: 1
- `count`: 1

## 4. 训练

### 4.1 DINOv3 ConvNeXt

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

### 4.2 DINOv3 ViT

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

说明：

- `MODEL.MTLORA.FREEZE_PRETRAINED=True` 时，主干参数默认冻结，仅训练 LoRA 和 decoder / task heads。
- ViT 分支会额外冻结更深层的 DPT 融合块，以匹配你当前实验设置。

## 5. 继续训练与验证

继续训练：

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

仅验证：

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

## 6. 输出内容

训练输出默认保存在：

```text
<output>/<model_name>/<tag>/
```

主要文件包括：

- `config.json`：本次运行的完整配置
- `log_rank0.txt`：训练日志
- `training_log.json`：按 epoch 保存的 loss / metric
- `mtl_loss_curve.png`：总损失曲线
- `task_loss_curves.png`：各任务损失曲线
- `ckpt_epoch_*.pth`：周期性 checkpoint

## 7. License

本仓库沿用根目录中的 [LICENSE](LICENSE)。
