import torch
import torch.nn as nn

from .convnext import ConvNeXt, convnext_sizes
from .vision_transformer import DinoVisionTransformer

class DINOv3ContextBackbone(nn.Module):
    """
    ConvNeXt-Small backbone wrapper for DINOv3-context pretrain weights.
    Returns 4-stage multi-scale features as spatial tensors.
    """

    def __init__(
        self,
        model_name="convnext_small",
        pretrained_path="",
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        # Build the official DINOv3 ConvNeXt architecture (checkpoint key compatible).
        if model_name not in {"convnext_tiny", "convnext_small", "convnext_base", "convnext_large"}:
            raise ValueError(f"Unsupported DINOv3 convnext backbone: {model_name}")
        size_key = model_name.replace("convnext_", "")
        if size_key not in convnext_sizes:
            raise ValueError(f"Unknown convnext size: {size_key}")
        cfg = convnext_sizes[size_key]

        self.model = ConvNeXt(
            in_chans=3,
            depths=cfg["depths"],
            dims=cfg["dims"],
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
        )
        self.out_channels = list(cfg["dims"])
        self.num_layers = 4
        self.embed_dim = cfg["dims"][0]
        self.patch_embed = None
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("state_dict")
                or ckpt.get("model")
                or ckpt.get("teacher")
                or ckpt
            )
        else:
            state_dict = ckpt
        # Strip wrappers if present (official DINO keys should match strict=True).
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            if k.startswith("backbone."):
                k = k[len("backbone.") :]
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=True)
        # When strict=True, missing/unexpected should both be empty.
        print(
            f"[DINOv3ContextBackbone] loaded {ckpt_path} strict=True, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def forward(self, x, return_stages=True, current_task=None):
        feats = self.model.get_intermediate_layers(x)  # low->high resolution order, each is (B,C,H,W)
        if return_stages:
            return feats
        return feats[-1]


class DINOv3ViTBackbone(nn.Module):
    """
    ViT backbone wrapper for DINOv3 pretrain weights.
    Returns multi-scale features as spatial tensors.
    """
    def __init__(
        self,
        model_name="vit_small",
        pretrained_path="",
        patch_size=16,
        out_indices=(2, 5, 8, 11),
    ):
        super().__init__()
        inferred_patch_size = self._infer_patch_size_from_checkpoint(pretrained_path) if pretrained_path else None
        if inferred_patch_size is not None and inferred_patch_size != patch_size:
            print(
                f"[DINOv3ViTBackbone] Override patch_size from {patch_size} to {inferred_patch_size} "
                f"to match checkpoint: {pretrained_path}"
            )
            patch_size = inferred_patch_size
        # ViT configurations
        vit_configs = {
            "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        }

        if model_name not in vit_configs:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        cfg = vit_configs[model_name]
        self.model = DinoVisionTransformer(
            patch_size=patch_size,
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            # Align architecture with released DINOv3 ViT checkpoints.
            # These checkpoints contain storage tokens, LayerScale gammas,
            # and qkv bias_mask buffers.
            n_storage_tokens=4,
            layerscale_init=1e-5,
            mask_k_bias=True,
        )

        self.out_channels = [cfg["embed_dim"]] * len(out_indices)
        self.out_indices = out_indices
        self.patch_size = patch_size
        self.embed_dim = cfg["embed_dim"]

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    @staticmethod
    def _infer_patch_size_from_checkpoint(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("teacher") or ckpt
            else:
                state_dict = ckpt

            patch_key = None
            for k in state_dict.keys():
                kk = k[7:] if k.startswith("module.") else k
                kk = kk[9:] if kk.startswith("backbone.") else kk
                if kk == "patch_embed.proj.weight":
                    patch_key = k
                    break
            if patch_key is None:
                return None

            w = state_dict[patch_key]
            if not torch.is_tensor(w) or w.dim() != 4:
                return None
            kh, kw = int(w.shape[-2]), int(w.shape[-1])
            if kh != kw:
                return None
            return kh
        except Exception as e:
            print(f"[DINOv3ViTBackbone] Failed to infer patch_size from checkpoint: {e}")
            return None

    def load_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("teacher") or ckpt
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(f"[DINOv3ViTBackbone] loaded {ckpt_path}, missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x, return_stages=True, current_task=None):
        # Get intermediate features from ViT
        B, C, H, W = x.shape
        features = self.model.get_intermediate_layers(x, n=self.out_indices, reshape=True)

        if return_stages:
            return features
        return features[-1]

