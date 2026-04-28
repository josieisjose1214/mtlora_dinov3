# --------------------------------------------------------
# PET 计数头：将 MTLoRA backbone 的多阶段特征送入 PET backbone 之后的组件，并复用 PET 损失
# --------------------------------------------------------
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证可导入 pet_models 与 util
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from util.misc import NestedTensor


def _stage_to_spatial(x, resolution):
    """(B, L, C) -> (B, C, H, W)"""
    B, L, C = x.shape
    H = W = resolution
    assert L == H * W
    return x.transpose(1, 2).view(B, C, H, W)


def _pet_enc_win_list_for_grid(sparse_res: int):
    """
    PET context encoder 的 progressive 窗口原为 112x112 设计 [(28,14)x2, (14,7)x2]。
    对任意 sparse_res（须被 16 整除，与 quadtree 一致），按比例缩放窗口使 H、W 均可整除。
    """
    s = sparse_res
    return [
        (28 * s // 112, 14 * s // 112),
        (28 * s // 112, 14 * s // 112),
        (14 * s // 112, 7 * s // 112),
        (14 * s // 112, 7 * s // 112),
    ]


class PETCountHead(nn.Module):
    """
    将 MTLoRA backbone 的 stage 特征转为 PET 所需的 4x/8x 特征，再跑 PET 的 context_encoder、
    quadtree_splitter、quadtree_sparse、quadtree_dense，输出与 PET 一致，用于 PET 的 SetCriterion。
    backbone_stage_list: list of (x, task_dict)，来自 Swin return_stages=True；
        每项 x 为 (B, L, C)，对应 4 个 stage 的 resolution 依次为 56, 28, 14, 7（448 输入）。
    为提升计数所需的空间细节：在 56x56 上融合 stage0+stage1（28x 上采样），再上采样到
    pet_res_sparse / pet_res_dense（默认 96/192，折中速度与精度；曾用 112/224 更准但更慢）。
    sparse_stride=4、dense_stride=2；建议保持 pet_res_dense == 2 * pet_res_sparse 与 quadtree 分支一致。
    pet_res_sparse 须被 16 整除（quadtree_splitter 使用 AvgPool 核 8x16）。
    """
    def __init__(
        self,
        embed_dim=96,
        stage0_dim=192,
        hidden_dim=256,
        num_classes=1,
        img_size=448,
        pet_res_sparse=96,
        pet_res_dense=192,
        use_single_task_recipe=False,
        args=None,
    ):
        super().__init__()
        if pet_res_sparse % 16 != 0:
            raise ValueError(
                f"pet_res_sparse={pet_res_sparse} must be divisible by 16 (quadtree AvgPool kernel 8x16)"
            )
        self.pet_res_sparse = pet_res_sparse
        self.pet_res_dense = pet_res_dense
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_single_task_recipe = use_single_task_recipe
        # Swin return_stages：list[0]/[1] 为各 stage 经 PatchMerging 后的通道，
        # 与 BasicLayer 内 dim 一致，即 C0=embed_dim*2、C1=embed_dim*4（以 EMBED_DIM=96 为例为 192+384）
        fuse_in = embed_dim * 2 + embed_dim * 4
        self.fuse = nn.Conv2d(fuse_in, stage0_dim, kernel_size=1)
        self.proj_8x = nn.Conv2d(stage0_dim, hidden_dim, 1)
        self.proj_4x = nn.Conv2d(stage0_dim, hidden_dim, 1)

        if args is None:
            from types import SimpleNamespace
            args = SimpleNamespace(
                hidden_dim=256,
                position_embedding='sine',
                dropout=0.0,
                nheads=8,
                dim_feedforward=512,
                enc_layers=4,
                dec_layers=2,
                set_cost_class=1.0,
                set_cost_point=1.0,
                ce_loss_coef=1.0,
                point_loss_coef=1.0,
                eos_coef=0.4,
            )

        from pet_models.position_encoding import build_position_encoding
        from pet_models.transformer.prog_win_transformer import build_encoder, build_decoder

        self.pos_embed = build_position_encoding(args)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        ])
        self.encode_feats = '8x'
        if self.use_single_task_recipe:
            # strict migration from single_task CountModel
            enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]
        else:
            enc_win_list = _pet_enc_win_list_for_grid(pet_res_sparse)
        args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)
        context_patch = (128, 64)
        context_w = context_patch[0] // 8
        context_h = context_patch[1] // 8
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        if self.use_single_task_recipe:
            args.sparse_stride, args.dense_stride = 8, 4
        else:
            # 与上采样后的特征网格对齐：sparse / dense 步长相对原 PET 448 设定不变
            args.sparse_stride, args.dense_stride = 4, 2
        transformer = build_decoder(args)
        from pet_models.pet import BasePETCount
        self.quadtree_sparse = BasePETCount(None, num_classes, 'sparse', args=args, transformer=transformer)
        self.quadtree_dense = BasePETCount(None, num_classes, 'dense', args=args, transformer=transformer)

    def _build_features_and_pos(self, feat_4x, feat_8x, device, dtype):
        """从 (B,C,H,W) 的 4x/8x 特征构建 PET 的 features 与 pos。"""
        B, C, H4, W4 = feat_4x.shape
        _, _, H8, W8 = feat_8x.shape
        mask_4x = torch.zeros((B, H4, W4), dtype=torch.bool, device=device)
        mask_8x = torch.zeros((B, H8, W8), dtype=torch.bool, device=device)
        nt_4x = NestedTensor(feat_4x, mask_4x)
        nt_8x = NestedTensor(feat_8x, mask_8x)
        features = {'4x': nt_4x, '8x': nt_8x}
        pos_4x = self.pos_embed(nt_4x).to(dtype)
        pos_8x = self.pos_embed(nt_8x).to(dtype)
        pos = {'4x': pos_4x, '8x': pos_8x}
        return features, pos

    def forward(self, backbone_stage_list, image_tensor, train=True):
        """
        backbone_stage_list: list of (x, task_dict)，x 为 (B, L, C)
        image_tensor: (B, 3, H, W)，用于 NestedTensor samples 与 point query 的尺寸
        train: 是否训练（影响 point query 的生成方式）
        返回: dict，包含 'sparse'/'dense'/'split_map_*'（供外部用 PET SetCriterion 算损失），或推理时为 'pred_logits'/'pred_points' 等
        """
        device = image_tensor.device
        dtype = image_tensor.dtype
        B, _, img_h, img_w = image_tensor.shape

        # 兼容两种输入:
        # 1) Swin stages: list[(x, task_dict)]，x 为 (B, L, C)
        # 2) DPT features: list[Tensor]，每项为 (B, C, H, W)
        if (
            isinstance(backbone_stage_list, (list, tuple))
            and len(backbone_stage_list) > 0
            and torch.is_tensor(backbone_stage_list[0])
            and backbone_stage_list[0].dim() == 4
        ):
            if len(backbone_stage_list) < 3:
                raise ValueError("DPT 特征模式下 PETCountHead 需要至少 3 个尺度特征")
            # 与 single_task 计数基线一致：feat_2 作为 4x，feat_3 作为 8x
            feat_4x = backbone_stage_list[1]
            feat_8x = backbone_stage_list[2]
            if feat_4x.shape[1] != self.hidden_dim:
                feat_4x = self.proj_4x(feat_4x)
            if feat_8x.shape[1] != self.hidden_dim:
                feat_8x = self.proj_8x(feat_8x)
            dpt_feature_mode = True
        else:
            # Swin 模式：融合 stage0(56x56) + stage1(28x28->56)
            stage0 = backbone_stage_list[0][0]
            if len(backbone_stage_list) < 2:
                raise ValueError("PETCountHead 需要至少 2 个 backbone stage 以融合多尺度特征")
            stage1 = backbone_stage_list[1][0]
            s0 = _stage_to_spatial(stage0, 56)
            s1 = _stage_to_spatial(stage1, 28)
            s1_up = F.interpolate(s1, size=(56, 56), mode='bilinear', align_corners=False)
            feat_core = self.fuse(torch.cat([s0, s1_up], dim=1))
            res_8, res_4 = self.pet_res_sparse, self.pet_res_dense
            feat_8x = self.proj_8x(
                F.interpolate(feat_core, size=(res_8, res_8), mode='bilinear', align_corners=False)
            )
            feat_4x = self.proj_4x(
                F.interpolate(feat_core, size=(res_4, res_4), mode='bilinear', align_corners=False)
            )
            dpt_feature_mode = False

        features, pos = self._build_features_and_pos(feat_4x, feat_8x, device, dtype)
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        samples = NestedTensor(image_tensor, torch.zeros((B, img_h, img_w), dtype=torch.bool, device=device))
        kwargs = {'dense_input_embed': self.pos_embed(samples), 'train': train}

        src, mask = features[self.encode_feats].decompose()
        # if self.use_single_task_recipe and dpt_feature_mode:
        #     h8, w8 = int(src.shape[-2]), int(src.shape[-1])
        #     # single_task recipe's first encoder window becomes (h=16, w=32) inside prog_win_transformer
        #     if (h8 % 16) != 0 or (w8 % 32) != 0:
        #         raise ValueError(
        #             f"Count(PET) single_task recipe requires feat_8x divisible by (16,32), got {h8}x{w8}. "
        #             f"Current IMG_SIZE={image_tensor.shape[-2]}x{image_tensor.shape[-1]} is incompatible. "
        #             f"Use an IMG_SIZE that yields compatible feat_8x (e.g. width multiple for ViT path), "
        #             f"or disable strict single_task recipe."
        #         )
        src_pos_embed = pos[self.encode_feats]
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)

        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        outputs = {}
        if train or (split_map_sparse > 0.5).sum() > 0:
            if self.use_single_task_recipe and dpt_feature_mode:
                kw_s = {
                    **kwargs,
                    'div': split_map_sparse.reshape(bs, sp_h, sp_w),
                    'dec_win_size': [16, 8],
                }
            else:
                kw_s = {
                    **kwargs,
                    'div': split_map_sparse.reshape(bs, sp_h, sp_w),
                    'dec_win_size': [8, 8],
                    # 与 encode_src / 8x 特征同高宽，避免 PET decoder 里 memory 与 query 窗口数不一致
                    'pq_grid_hw': self.pet_res_sparse,
                }
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kw_s)
        else:
            outputs_sparse = None
        if train or (split_map_dense > 0.5).sum() > 0:
            if self.use_single_task_recipe and dpt_feature_mode:
                kw_d = {
                    **kwargs,
                    'div': split_map_dense.reshape(bs, ds_h, ds_w),
                    'dec_win_size': [8, 4],
                }
            else:
                kw_d = {
                    **kwargs,
                    'div': split_map_dense.reshape(bs, ds_h, ds_w),
                    'dec_win_size': [8, 4],
                    'pq_grid_hw': self.pet_res_dense,
                }
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kw_d)
        else:
            outputs_dense = None

        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense

        if not train:
            # 推理：合并 sparse/dense 预测
            out_dense, out_sparse = outputs['dense'], outputs['sparse']
            thrs = 0.5
            index_sparse = None
            index_dense = None
            pred_count_per_image = torch.zeros(B, dtype=torch.float32, device=device)
            if outputs['sparse'] is not None:
                out_sparse_scores = F.softmax(out_sparse['pred_logits'], -1)[..., 1]
                index_sparse = (out_sparse_scores > thrs).cpu()
                pred_count_per_image += (out_sparse_scores > thrs).sum(dim=1).to(torch.float32)
            if outputs['dense'] is not None:
                out_dense_scores = F.softmax(out_dense['pred_logits'], -1)[..., 1]
                index_dense = (out_dense_scores > thrs).cpu()
                pred_count_per_image += (out_dense_scores > thrs).sum(dim=1).to(torch.float32)
            div_out = {}
            output_names = (out_sparse or out_dense).keys()
            for name in list(output_names):
                if 'pred' in name:
                    if index_dense is None:
                        div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                    elif index_sparse is None:
                        div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                    else:
                        div_out[name] = torch.cat([
                            out_sparse[name][index_sparse].unsqueeze(0),
                            out_dense[name][index_dense].unsqueeze(0),
                        ], dim=1)
                else:
                    div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
            div_out['split_map_raw'] = outputs['split_map_raw']
            # 用于评估阶段按单图统计计数指标（MAE / R^2）
            div_out['pred_count_per_image'] = pred_count_per_image
            return div_out
        return outputs
