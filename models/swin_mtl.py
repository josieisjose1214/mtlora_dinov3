# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import types


def get_head(task, backbone_channels, num_outputs, config=None, multiscale=True):
    """Return the decoder head"""
    head_type = config.MODEL.DECODER_HEAD.get(task, "hrnet")

    if head_type == "hrnet":
        print(
            f"Using hrnet for task {task} with backbone channels {backbone_channels}")
        from models.seg_hrnet import HighResolutionHead

        return HighResolutionHead(backbone_channels, num_outputs)
        
    elif head_type == "yolo":
        print(f"Using yolov8 head for task {task}")#默认检测任务使用
        from models.yolo_head import YOLOHead

        return YOLOHead(num_outputs)
    elif head_type == "fcos":
        print(f"Using FCOS head for task {task}")
        from models.fcos_head import FCOSHead

        return FCOSHead(in_channels=256, num_classes=num_outputs, num_levels=4)

    elif head_type == "simplenet":
        print(f"Using simple classification head for task {task}")#默认分类任务使用
        from models.classify_head import ClassificationHead

        return ClassificationHead(144,num_outputs) #只使用最后一层的输出特征图
        
    elif head_type == "updecoder":
        print(f"Using updecoder for task {task}")
        from models.updecoder import Decoder

        return Decoder(
            backbone_channels,
            num_outputs,
            args=types.SimpleNamespace(
                **{
                    "num_deconv": 3,
                    "num_filters": [32, 32, 32],
                    "deconv_kernels": [2, 2, 2],
                }
            ),
        )
    elif head_type == "segformer":
        print(
            f"Using segformer for task {task} with {config.MODEL.SEGFORMER_CHANNELS} channels"
        )
        from models.segformer import SegFormerHead

        return SegFormerHead(
            in_channels=backbone_channels,
            channels=config.MODEL.SEGFORMER_CHANNELS,
            num_classes=num_outputs,
        )
    elif head_type == "pet" or task == "count":
        print(f"Using PET count head for task {task}")
        from models.pet_head import PETCountHead
        img_size = config.DATA.IMG_SIZE if hasattr(config, 'DATA') and hasattr(config.DATA, 'IMG_SIZE') else 448
        # backbone_stage_list[0] 的通道：Swin 首层输出为 embed_dim，但实际 return 的 stage0 多为第二层维度 192（embed_dim*2）
        embed = config.MODEL.SWIN.EMBED_DIM
        stage0_dim = getattr(config.MODEL.SWIN, 'STAGE0_DIM', embed * 2)
        return PETCountHead(
            embed_dim=embed,
            stage0_dim=stage0_dim,
            hidden_dim=256,
            num_classes=1,
            img_size=img_size,
            pet_res_sparse=config.MODEL.PET_COUNT_RES_SPARSE,
            pet_res_dense=config.MODEL.PET_COUNT_RES_DENSE,
        )
    else:
        if not multiscale:
            from models.aspp_single import DeepLabHead
        else:
            from models.aspp import DeepLabHead
        print(f"Using ASPP for task {task}")
        return DeepLabHead(backbone_channels, num_outputs)


class DecoderGroup(nn.Module):
    def __init__(self, tasks, num_outputs, channels, out_size, config, multiscale=True):
        super(DecoderGroup, self).__init__()
        self.tasks = tasks
        self.num_outputs = num_outputs
        self.channels = channels
        self.decoders = nn.ModuleDict()
        self.out_size = out_size
        self.multiscale = multiscale
        for task in self.tasks:
            self.decoders[task] = get_head(
                task,
                self.channels,
                self.num_outputs[task],
                config=config,
                multiscale=self.multiscale,
            )

    def forward(self, x, image=None):
        # x 可能只包含当前任务（当传入 current_task 时），只对 x 中存在的任务跑 Decoder
        # 计数任务 (count) 时 x[task] 为 backbone 的 stage 列表，且需传入 image
        result = {}
        for task in x:
            if task == "count" and image is not None:
                decoder_output = self.decoders[task](x[task], image, train=self.training)
            else:
                decoder_output = self.decoders[task](x[task])
            if task == "classify":
                result[task] = decoder_output
            elif task == "detect":
                result[task] = decoder_output
            elif task == "count":
                result[task] = decoder_output
            else:
                result[task] = F.interpolate(
                    decoder_output, self.out_size, mode="bilinear"
                )
        return result


class Downsampler(nn.Module):
    def __init__(self, dims, channels, input_res, bias=False, enabled=True):
        super(Downsampler, self).__init__()
        self.dims = dims
        self.input_res = input_res
        self.enabled = enabled
        if self.enabled:
            self.downsample_0 = torch.nn.Conv2d(
                dims[0], channels[0], 1, bias=bias)
            self.downsample_1 = torch.nn.Conv2d(
                dims[1], channels[1], 1, bias=bias)
            self.downsample_2 = torch.nn.Conv2d(
                dims[2], channels[2], 1, bias=bias)
            self.downsample_3 = torch.nn.Conv2d(
                dims[3], channels[3], 1, bias=bias)

    def forward(self, x):
        s_3 = (
            x[3]
            .view(-1, self.input_res[3], self.input_res[3], self.dims[3])
            .permute(0, 3, 1, 2)
        )

        s_2 = (
            x[2]
            .view(-1, self.input_res[2], self.input_res[2], self.dims[2])
            .permute(0, 3, 1, 2)
        )
        s_1 = (
            x[1]
            .view(-1, self.input_res[1], self.input_res[1], self.dims[1])
            .permute(0, 3, 1, 2)
        )
        s_0 = (
            x[0]
            .view(-1, self.input_res[0], self.input_res[0], self.dims[0])
            .permute(0, 3, 1, 2)
        )

        if self.enabled:
            return [
                self.downsample_0(s_0),
                self.downsample_1(s_1),
                self.downsample_2(s_2),
                self.downsample_3(s_3),
            ]
        else:
            return [s_0, s_1, s_2, s_3]


class MultiTaskSwin(nn.Module):
    def __init__(self, encoder, config):
        super(MultiTaskSwin, self).__init__()

        self.backbone = encoder
        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.tasks = config.TASKS
        if hasattr(self.backbone, "patch_embed"):
            patches_resolution = self.backbone.patch_embed.patches_resolution
            self.embed_dim = self.backbone.embed_dim
            num_layers = self.backbone.num_layers
            self.dims = [
                int((self.embed_dim * 2 ** ((i + 1) if i < num_layers - 1 else i)))
                for i in range(num_layers)
            ]
            self.input_res = [
                patches_resolution[0] // (2 **
                                          ((i + 1) if i < num_layers - 1 else i))
                for i in range(num_layers)
            ]
            self.window_size = self.backbone.layers[0].blocks[0].window_size
            self.img_size = self.backbone.patch_embed.img_size
        else:
            self.input_res = [28, 14, 7, 7]

            self.dims = [192, 384, 768, 768]
            self.window_size = config.MODEL.SWIN.WINDOW_SIZE
            self.img_size = config.DATA.IMG_SIZE

        self.channels = (
            config.MODEL.DECODER_CHANNELS
            if config.MODEL.DECODER_DOWNSAMPLER
            else self.dims
        )
        self.mtlora = config.MODEL.MTLORA
        if self.mtlora.ENABLED:
            self.downsampler = nn.ModuleDict(
                {
                    task: Downsampler(
                        dims=self.dims,
                        channels=self.channels,
                        input_res=self.input_res,
                        bias=False,
                    )
                    for task in self.tasks
                }
            )
        else:
            self.downsampler = Downsampler(
                dims=self.dims,
                channels=self.channels,
                input_res=self.input_res,
                bias=False,
            )

        self.per_task_downsampler = config.MODEL.PER_TASK_DOWNSAMPLER
        if self.per_task_downsampler:
            self.downsampler = nn.ModuleDict(
                {
                    task: Downsampler(
                        dims=self.dims,
                        channels=self.channels,
                        input_res=self.input_res,
                        bias=False,
                        enabled=config.MODEL.DECODER_DOWNSAMPLER,
                    )
                    for task in self.tasks
                }
            )
        else:
            self.downsampler = Downsampler(
                dims=self.dims,
                channels=self.channels,
                input_res=self.input_res,
                bias=False,
            )
        self.decoders = DecoderGroup(
            self.tasks,
            self.num_outputs,
            channels=self.channels,
            out_size=self.img_size,
            config=config,
            multiscale=True,
        )

    def forward(self, x, current_task=None):
        # 传递current_task给backbone，只计算当前任务的LoRA
        shared_representation = self.backbone(x, return_stages=True, current_task=current_task)

        if self.mtlora.ENABLED:
            # 如果指定了current_task，只处理当前任务
            if current_task is not None and current_task in self.tasks:
                if current_task == 'count':
                    # 计数任务：不经过 downsampler，直接传 backbone 的 stage 列表给 PET 头
                    shared_ft = {current_task: shared_representation}
                else:
                    shared_ft = {current_task: []}
                    for _, tasks_shared_rep in shared_representation:
                        if current_task in tasks_shared_rep:
                            shared_ft[current_task].append(tasks_shared_rep[current_task])
                    shared_ft[current_task] = self.downsampler[current_task](shared_ft[current_task])
            else:
                # 处理所有任务（兼容模式）
                shared_ft = {task: [] for task in self.tasks}
                for _, tasks_shared_rep in shared_representation:
                    for task, shared_rep in tasks_shared_rep.items():
                        shared_ft[task].append(shared_rep)
                for task in self.tasks:
                    if task == 'count':
                        shared_ft[task] = shared_representation
                    else:
                        shared_ft[task] = self.downsampler[task](shared_ft[task])
        else:
            if self.per_task_downsampler:
                if current_task is not None and current_task in self.tasks:
                    if current_task == 'count':
                        shared_ft = {current_task: shared_representation}
                    else:
                        shared_ft = {current_task: self.downsampler[current_task](shared_representation)}
                else:
                    shared_ft = {}
                    for task in self.tasks:
                        shared_ft[task] = shared_representation if task == 'count' else self.downsampler[task](shared_representation)
            else:
                shared_representation = self.downsampler(shared_representation)
                shared_ft = {
                    task: shared_representation for task in self.tasks}

        image_for_count = x if (current_task == 'count' or 'count' in shared_ft) else None
        result = self.decoders(shared_ft, image=image_for_count)
        return result

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = False

    def unfreeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
