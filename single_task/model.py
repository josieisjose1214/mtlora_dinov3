from torch import nn
import torch.nn.functional as F

from dinov3.backbone import DINOv3ContextBackbone, DINOv3ViTBackbone
from dpt_blocks import FeatureFusionBlock_custom, Slice, Transpose
from pet_head.pet import BasePETCount, SetCriterion
from pet_head.transformer import build_encoder, build_decoder
from pet_head.position_encoding import build_position_encoding
from pet_head.matcher import build_matcher
from pet_head.misc import NestedTensor, nested_tensor_from_tensor_list
from fcos_head import FCOSDetector
import torch


class Args:
    """Arguments for PET head"""
    def __init__(self):
        self.hidden_dim = 256
        self.position_embedding = 'sine'
        self.enc_layers = 4
        self.dec_layers = 2
        self.dim_feedforward = 512
        self.dropout = 0.0
        self.nheads = 8
        self.num_feature_levels = 1
        self.sparse_stride = 8
        self.dense_stride = 4
        self.ce_loss_coef = 1.0
        self.point_loss_coef = 5.0
        self.eos_coef = 0.5
        # Matcher parameters
        self.set_cost_class = 1.0
        self.set_cost_point = 0.05


class BackboneDecoder(nn.Module):
    def __init__(self, model_name, path, head=None):
        super(BackboneDecoder, self).__init__()

        # Determine backbone type
        if 'vit' in model_name:
            self.backbone = DINOv3ViTBackbone(model_name=model_name, pretrained_path=path)
            self.backbone_type = 'vit'
            vit_features = self.backbone.embed_dim
            features = [96, 192, 384, 768]

            # Reassemble modules for ViT (following DPT official implementation)
            # Output sizes: feat_1=64x64, feat_2=32x32, feat_3=16x16, feat_4=8x8
            self.reassemble1 = nn.Sequential(
                nn.Conv2d(vit_features, features[0], kernel_size=1),
                nn.ConvTranspose2d(features[0], features[0], kernel_size=4, stride=4),
            )
            self.reassemble2 = nn.Sequential(
                nn.Conv2d(vit_features, features[1], kernel_size=1),
                nn.ConvTranspose2d(features[1], features[1], kernel_size=2, stride=2),
            )
            self.reassemble3 = nn.Sequential(
                nn.Conv2d(vit_features, features[2], kernel_size=1),
            )
            self.reassemble4 = nn.Sequential(
                nn.Conv2d(vit_features, features[3], kernel_size=1),
                nn.Conv2d(features[3], features[3], kernel_size=3, stride=2, padding=1),
            )
            backbone_channels = features
        else:
            self.backbone = DINOv3ContextBackbone(model_name=model_name, pretrained_path=path)
            self.backbone_type = 'convnext'
            backbone_channels = self.backbone.out_channels

        hidden_dim = 256

        # Projection layers
        self.proj1 = nn.Conv2d(backbone_channels[0], hidden_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(backbone_channels[1], hidden_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(backbone_channels[2], hidden_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(backbone_channels[3], hidden_dim, kernel_size=1)

        # DPT fusion blocks
        self.fusion1 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))
        self.fusion2 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))
        self.fusion3 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))
        self.fusion4 = FeatureFusionBlock_custom(hidden_dim, nn.ReLU(False))

        self.output_conv = head

    def forward(self, x):
        if self.backbone_type == 'vit':
            # ViT: get intermediate layer outputs (B, N, C) format
            B, C, H, W = x.shape
            #print(f'[DEBUG] Input shape:{x.shape},H={H},W={W}')
            patch_size = self.backbone.patch_size
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            #print(f"[DEBUG] Patch size:{patch_size},num_patches:{num_patches_h}x{num_patches_w}")

            # Get features in token format (B, N+1, C)
            layers = self.backbone.model.get_intermediate_layers(x, n=self.backbone.out_indices, reshape=False)
            #print(f"[DEBUG] Layer 0 shape (tokens): {layers[0].shape}")

            # Manually reshape: remove CLS token and reshape to spatial
            feat_1 = self._reshape_vit_tokens(layers[0], num_patches_h, num_patches_w)
            feat_2 = self._reshape_vit_tokens(layers[1], num_patches_h, num_patches_w)
            feat_3 = self._reshape_vit_tokens(layers[2], num_patches_h, num_patches_w)
            feat_4 = self._reshape_vit_tokens(layers[3], num_patches_h, num_patches_w)

            # Apply reassemble
            feat_1 = self._apply_reassemble_ops(feat_1, self.reassemble1)
            feat_2 = self._apply_reassemble_ops(feat_2, self.reassemble2)
            feat_3 = self._apply_reassemble_ops(feat_3, self.reassemble3)
            feat_4 = self._apply_reassemble_ops(feat_4, self.reassemble4)
        else:
            # ConvNeXt: already in (B, C, H, W) format
            feat_1, feat_2, feat_3, feat_4 = self.backbone.model.get_intermediate_layers(x, n=4, reshape=True)

        # Project features to hidden_dim
        feat_1 = self.proj1(feat_1)
        feat_2 = self.proj2(feat_2)
        feat_3 = self.proj3(feat_3)
        feat_4 = self.proj4(feat_4)

        # DPT fusion
        feat_4 = self.fusion4(feat_4)
        feat_3 = self.fusion3(feat_4, feat_3)
        feat_2 = self.fusion2(feat_3, feat_2)
        feat_1 = self.fusion1(feat_2, feat_1)

        if self.output_conv is not None:
            out = self.output_conv(feat_1)
            return out

        return [feat_1, feat_2, feat_3, feat_4]

    def _reshape_vit_tokens(self, tokens, h, w):
        """Reshape ViT tokens (B, N+1, C) to spatial format (B, C, H, W)"""
        B, N_total, C = tokens.shape
        expected_N = h * w

        # DINOv3 may have register tokens, keep only spatial tokens
        # Assume first token is CLS, last N tokens are spatial patches
        tokens = tokens[:, -expected_N:]  # Take last h*w tokens

        # Reshape to spatial
        tokens = tokens.transpose(1, 2)  # (B, C, N)
        tokens = tokens.reshape(B, C, h, w)  # (B, C, h, w)
        return tokens

    def _apply_reassemble_ops(self, tokens, reassemble_module):
        """Apply only conv and upsample operations (skip Slice/Transpose/Unflatten)"""
        for module in reassemble_module:
            if isinstance(module, (Slice, Transpose, nn.Unflatten)):
                continue
            tokens = module(tokens)
        return tokens

    def _reassemble_vit(self, tokens, h, w, reassemble_module):
        """Apply reassemble with dynamic spatial size"""
        # tokens: (B, N+1, C) where N = h*w
        B, N_plus_1, C = tokens.shape

        # Remove CLS token
        tokens = tokens[:, 1:]  # (B, N, C)
        N = tokens.shape[1]

        # Verify dimensions match
        assert N == h * w, f"Token count mismatch: got {N} tokens but expected {h}x{w}={h*w}"

        # Transpose and reshape
        tokens = tokens.transpose(1, 2)  # (B, C, N)
        tokens = tokens.reshape(B, C, h, w)  # (B, C, h, w)

        # Apply conv and upsample layers (skip Slice/Transpose/Unflatten)
        for module in reassemble_module:
            if isinstance(module, (Slice, Transpose, nn.Unflatten)):
                continue
            tokens = module(tokens)

        return tokens


class CountModel(nn.Module):
    def __init__(self, model_name="convnext_small", pretrained_path="dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"):
        super(CountModel, self).__init__()
        self.args = Args()

        # Backbone + Decoder
        self.backbone_decoder = BackboneDecoder(model_name, pretrained_path)

        # Freeze backbone weights
        for param in self.backbone_decoder.backbone.parameters():
            param.requires_grad = False

        # For ViT: freeze bottom fusion blocks and reassemble to reduce overfitting
        if 'vit' in model_name:
            for param in self.backbone_decoder.fusion3.parameters():
                param.requires_grad = False
            for param in self.backbone_decoder.fusion4.parameters():
                param.requires_grad = False
            # Note: reassemble modules are just Conv layers, already included in fusion freezing

        # Position encoding
        self.pos_embed = build_position_encoding(self.args)

        # Feature projection (256 -> 256, already at correct dim)
        hidden_dim = self.args.hidden_dim
        self.input_proj_4x = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.input_proj_8x = nn.Conv2d(256, hidden_dim, kernel_size=1)

        # Context encoder
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]
        self.args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(self.args, enc_win_list=enc_win_list)

        # Quadtree splitter
        context_patch = (128, 64)
        context_w = context_patch[0] // 8
        context_h = context_patch[1] // 8
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # Point-query quadtree layers
        num_classes = 1
        transformer = build_decoder(self.args)
        self.quadtree_sparse = BasePETCount(
            None, num_classes, quadtree_layer='sparse',
            args=self.args, transformer=transformer
        )
        self.quadtree_dense = BasePETCount(
            None, num_classes, quadtree_layer='dense',
            args=self.args, transformer=build_decoder(self.args)
        )

    def forward(self, samples, **kwargs):
        # Convert to NestedTensor if needed
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Get fused features from backbone+decoder
        x = samples.tensors
        feats = self.backbone_decoder(x)  # [feat_1, feat_2, feat_3, feat_4]

        # Create mask for features
        mask = samples.mask
        if mask is None:
            mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]),
                             dtype=torch.bool, device=x.device)

        # Prepare features dict (use feat_2 as 4x, feat_3 as 8x)
        feat_4x = feats[1]  # Higher resolution
        feat_8x = feats[2]  # Lower resolution

        # Downsample mask to match feature sizes
        mask_4x = F.interpolate(mask.float().unsqueeze(1),
                               size=feat_4x.shape[-2:]).squeeze(1).bool()
        mask_8x = F.interpolate(mask.float().unsqueeze(1),
                               size=feat_8x.shape[-2:]).squeeze(1).bool()

        # Project features
        feat_4x = self.input_proj_4x(feat_4x)
        feat_8x = self.input_proj_8x(feat_8x)

        # Create features dict
        features = {
            '4x': NestedTensor(feat_4x, mask_4x),
            '8x': NestedTensor(feat_8x, mask_8x)
        }

        # Position encoding for dense input
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # Position encoding for each feature scale
        pos = {
            '4x': self.pos_embed(NestedTensor(feat_4x, mask_4x)),
            '8x': self.pos_embed(NestedTensor(feat_8x, mask_8x))
        }

        # Forward through PET head
        if 'train' in kwargs:
            return self.train_forward(samples, features, pos, **kwargs)
        else:
            return self.test_forward(samples, features, pos, **kwargs)

    def pet_forward(self, samples, features, pos, **kwargs):
        """PET head forward pass"""
        # Context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)

        # Quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        # Sparse layer
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None

        # Dense layer
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None

        return {
            'sparse': outputs_sparse,
            'dense': outputs_dense,
            'split_map_raw': split_map,
            'split_map_sparse': split_map_sparse,
            'split_map_dense': split_map_dense
        }

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """Compute loss for training"""
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # Compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # Sparse loss
        loss_dict_sparse = {k+'_sp': v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp': v for k, v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k]
                            for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # Dense loss
        loss_dict_dense = {k+'_ds': v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds': v for k, v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k]
                           for k in loss_dict_dense.keys() if k in weight_dict_dense)

        losses = loss_pq_sparse + loss_pq_dense

        # Update dicts
        loss_dict = {**loss_dict_sparse, **loss_dict_dense}
        weight_dict = {**weight_dict_sparse, **weight_dict_dense}

        # Quadtree splitter loss
        den = torch.tensor([target['density'] for target in targets])
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()
        if sum(ds_idx) > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        loss_split = loss_split_sp + loss_split_ds
        weight_split = 0.1 if epoch >= warmup_ep else 0.0
        loss_dict['loss_split'] = loss_split
        weight_dict['loss_split'] = weight_split
        losses += loss_split * weight_split

        return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}

    def train_forward(self, samples, features, pos, **kwargs):
        """Training forward pass"""
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses

    def test_forward(self, samples, features, pos, **kwargs):
        """Testing forward pass"""
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        thrs = 0.5

        # Process sparse
        if out_sparse is not None:
            out_sparse_scores = F.softmax(out_sparse['pred_logits'], -1)[..., 1]
            index_sparse = (out_sparse_scores > thrs).cpu()
        else:
            index_sparse = None

        # Process dense
        if out_dense is not None:
            out_dense_scores = F.softmax(out_dense['pred_logits'], -1)[..., 1]
            index_dense = (out_dense_scores > thrs).cpu()
        else:
            index_dense = None

        # Format output
        div_out = {}
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([
                        out_sparse[name][index_sparse].unsqueeze(0),
                        out_dense[name][index_dense].unsqueeze(0)
                    ], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        return div_out

class DetectModel(nn.Module):
    def __init__(self, model_name="convnext_small", pretrained_path="dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth", num_classes=1):
        super(DetectModel, self).__init__()

        # Backbone + Decoder (no head, returns 4 feature levels)
        self.backbone_decoder = BackboneDecoder(model_name, pretrained_path, head=None)

        # Freeze backbone
        for param in self.backbone_decoder.backbone.parameters():
            param.requires_grad = False

        # Freeze bottom fusion blocks
        for param in self.backbone_decoder.fusion3.parameters():
            param.requires_grad = False
        for param in self.backbone_decoder.fusion4.parameters():
            param.requires_grad = False

        # FCOS detector head (uses 4 FPN levels with strides [4,8,16,32])
        self.detector = FCOSDetector(in_channels=256, num_classes=num_classes, fpn_strides=[4, 8, 16, 32])

    def forward(self, x, targets=None):
        # Get 4-level feature pyramid from DPT decoder
        feats = self.backbone_decoder(x)  # [feat_1, feat_2, feat_3, feat_4]
        # feats are at strides [2, 4, 8, 16] after fusion upsampling
        # Use feat_2, feat_3, feat_4 + add P5 for detection

        # Use last 3 levels + add P5 via stride-2 conv
        fpn_feats = feats[1:]  # [feat_2@stride4, feat_3@stride8, feat_4@stride16]
        p5 = F.max_pool2d(feats[3], kernel_size=2, stride=2)  # stride32
        fpn_feats.append(p5)

        return self.detector(fpn_feats, targets)

class SegmentModel(nn.Module):
    def __init__(self, model_name="convnext_small", pretrained_path="dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth", num_classes=4):
        super(SegmentModel, self).__init__()

        features = 256
        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        # Backbone + Decoder
        self.backbone_decoder = BackboneDecoder(model_name, pretrained_path, head)

        # Freeze backbone
        for param in self.backbone_decoder.backbone.parameters():
            param.requires_grad = False

        # Freeze bottom fusion blocks (fusion3, fusion4)
        for param in self.backbone_decoder.fusion3.parameters():
            param.requires_grad = False
        for param in self.backbone_decoder.fusion4.parameters():
            param.requires_grad = False

        # Unfreeze top fusion blocks (fusion1, fusion2) for fine-tuning
        for param in self.backbone_decoder.fusion1.parameters():
            param.requires_grad = True
        for param in self.backbone_decoder.fusion2.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone_decoder(x)


class ClassifyModel(nn.Module):
    def __init__(self, model_name="convnext_small", pretrained_path="dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth", num_classes=8):
        super(ClassifyModel, self).__init__()

        features = 256

        # Backbone + Decoder (no head, returns feature maps)
        self.backbone_decoder = BackboneDecoder(model_name, pretrained_path, head=None)

        # Freeze backbone
        for param in self.backbone_decoder.backbone.parameters():
            param.requires_grad = False

        # Freeze bottom fusion blocks
        for param in self.backbone_decoder.fusion3.parameters():
            param.requires_grad = False
        for param in self.backbone_decoder.fusion4.parameters():
            param.requires_grad = False

        # GAP + MLP classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features, features),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(features, num_classes),
        )

    def forward(self, x):
        feats = self.backbone_decoder(x)  # [feat_1, feat_2, feat_3, feat_4]
        # Use feat_1 (highest resolution fused feature) for classification
        out = self.classifier(feats[0])
        return out


def build_model(args):
    """Build model and criterion"""
    model_name = getattr(args, 'model_name', 'convnext_small')
    pretrained_path = getattr(args, 'pretrained_path', 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth')

    model = CountModel(model_name=model_name, pretrained_path=pretrained_path)

    # Build criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )

    return model, criterion
        