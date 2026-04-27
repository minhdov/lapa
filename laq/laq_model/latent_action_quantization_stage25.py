from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from laq_model.attention import Transformer, ContinuousPositionBias


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class LatentActionQuantizationStage25(nn.Module):
    """
    Stage 2.5 module for LAPA-depth.

    Goal:
        depth1 + z_rgb_indices -> z_depth_indices

    Inputs:
        depth1:        [B, C, H, W]
        z_rgb_indices: [B, code_seq_len]

    Optional target:
        z_depth_indices: [B, code_seq_len]

    Outputs:
        If z_depth_indices is given:
            loss, z_refined_logits, z_refined_feature
        Else:
            z_refined_logits, z_refined_feature

    Typical config:
        dim=1024
        codebook_size=8
        image_size=256
        patch_size=32
        spatial_depth=8
        code_seq_len=4
    """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        dim_head=64,
        heads=8,
        channels=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        code_seq_len=4,
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)

        patch_height, patch_width = self.patch_size
        image_height, image_width = self.image_size

        assert image_height % patch_height == 0
        assert image_width % patch_width == 0

        self.spatial_rel_pos_bias = ContinuousPositionBias(
            dim=dim,
            heads=heads,
        )

        self.to_patch_emb_depth = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b h w (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )

        self.depth_spatial_transformer = Transformer(
            depth=spatial_depth,
            **transformer_kwargs,
        )

        # Embedding for z_rgb_indices, e.g. [B, 4] with values in [0, 7].
        self.z_rgb_embed = nn.Embedding(codebook_size, dim)

        # Fusion: depth geometry feature + RGB/text latent prior feature.
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Predict one distribution over codebook IDs for each latent slot.
        self.head = nn.Linear(dim, codebook_size)

    @property
    def patch_height_width(self):
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def load(self, path, strict=False):
        """
        Load a Stage 2.5 checkpoint.

        This is intended for checkpoints saved from this class.
        If you want to initialize partially from another model, load manually
        with strict=False and inspect missing/unexpected keys.
        """
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"

        pt = torch.load(str(path), map_location="cpu")

        if isinstance(pt, dict) and "model" in pt:
            pt = pt["model"]

        pt = {
            k.replace("module.", "") if "module." in k else k: v
            for k, v in pt.items()
        }

        return self.load_state_dict(pt, strict=strict)

    def encode_depth1(self, depth1):
        """
        Encode current depth frame into depth features.

        Args:
            depth1: [B, C, H, W]

        Returns:
            depth_feature: [B, D]
            depth_tokens:  [B, H_patch, W_patch, D]
        """
        assert depth1.ndim == 4, f"Expected [B, C, H, W], got {depth1.shape}"

        b, c, h_img, w_img = depth1.shape
        assert (h_img, w_img) == self.image_size, (
            f"Expected image size {self.image_size}, got {(h_img, w_img)}"
        )

        h, w = self.patch_height_width

        # [B, C, H, W] -> [B, H_patch, W_patch, D]
        depth_tokens = self.to_patch_emb_depth(depth1)

        # Add fake temporal dimension T=1.
        # [B, H, W, D] -> [B, 1, H, W, D]
        depth_tokens = rearrange(depth_tokens, "b h w d -> b 1 h w d")

        # Transformer expects video_shape = (B, T, H, W)
        video_shape = tuple(depth_tokens.shape[:-1])  # (B, 1, H, W)

        # [B, 1, H, W, D] -> [(B*T), H*W, D]
        tokens = rearrange(depth_tokens, "b t h w d -> (b t) (h w) d")

        attn_bias = self.spatial_rel_pos_bias(
            h,
            w,
            device=tokens.device,
        )

        tokens = self.depth_spatial_transformer(
            tokens,
            attn_bias=attn_bias,
            video_shape=video_shape,
        )

        # [(B*T), H*W, D] -> [B, T, H, W, D] -> [B, H, W, D]
        depth_tokens = rearrange(
            tokens,
            "(b t) (h w) d -> b t h w d",
            b=b,
            h=h,
            w=w,
        )[:, 0]

        # Global depth geometry feature.
        depth_feature = depth_tokens.mean(dim=(1, 2))  # [B, D]

        return depth_feature, depth_tokens

    def encode_z_rgb_indices(self, z_rgb_indices):
        """
        Convert z_rgb_indices into latent prior features.

        Args:
            z_rgb_indices: [B, code_seq_len]

        Returns:
            z_rgb_tokens:  [B, code_seq_len, D]
            z_rgb_feature: [B, D]
        """
        assert z_rgb_indices.ndim == 2, (
            f"Expected [B, code_seq_len], got {z_rgb_indices.shape}"
        )
        assert z_rgb_indices.shape[1] == self.code_seq_len, (
            f"Expected code_seq_len={self.code_seq_len}, "
            f"got {z_rgb_indices.shape[1]}"
        )

        z_rgb_indices = z_rgb_indices.long()
        z_rgb_tokens = self.z_rgb_embed(z_rgb_indices)  # [B, L, D]
        z_rgb_feature = z_rgb_tokens.mean(dim=1)        # [B, D]

        return z_rgb_tokens, z_rgb_feature

    def forward(
        self,
        depth1,
        z_rgb_indices,
        z_depth_indices=None,
    ):
        """
        Stage 2.5 forward.

        Args:
            depth1:          [B, C, H, W]
            z_rgb_indices:   [B, code_seq_len]
            z_depth_indices: [B, code_seq_len], optional

        Returns:
            If z_depth_indices is provided:
                loss, z_refined_logits, z_refined_feature
            Else:
                z_refined_logits, z_refined_feature
        """
        depth_feature, depth_tokens = self.encode_depth1(depth1)
        z_rgb_tokens, z_rgb_feature = self.encode_z_rgb_indices(z_rgb_indices)

        fused = torch.cat([depth_feature, z_rgb_feature], dim=-1)  # [B, 2D]
        z_refined_feature = self.fusion(fused)                     # [B, D]

        # One refined feature per latent slot.
        z_refined_tokens = z_refined_feature[:, None, :].repeat(
            1,
            self.code_seq_len,
            1,
        )  # [B, L, D]

        z_refined_logits = self.head(z_refined_tokens)  # [B, L, codebook_size]

        if z_depth_indices is not None:
            loss = F.cross_entropy(
                z_refined_logits.reshape(-1, z_refined_logits.shape[-1]),
                z_depth_indices.reshape(-1).long(),
            )
            return loss, z_refined_logits, z_refined_feature

        return z_refined_logits, z_refined_feature

    def predict_indices(self, depth1, z_rgb_indices):
        """
        Convenience function for inference.

        Returns:
            z_pred_indices: [B, code_seq_len]
        """
        z_refined_logits, _ = self.forward(
            depth1=depth1,
            z_rgb_indices=z_rgb_indices,
            z_depth_indices=None,
        )
        return z_refined_logits.argmax(dim=-1)
