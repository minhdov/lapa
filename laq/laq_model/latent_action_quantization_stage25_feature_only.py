from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LatentActionQuantizationStage252(nn.Module):
    """
    Stage 2.5.2 module for LAPA-depth.

    Pipeline:
        z_rgb_features -> z_depth_indices

    Difference from LatentActionQuantizationStage25:
        - Does NOT use depth1.
        - Does NOT include depth patch embedding or depth spatial transformer.
        - Uses only precomputed LAPA RGB features as input.

    Inputs:
        z_rgb_features: [B, z_rgb_feature_dim], e.g. [B, 4096]

    Optional target:
        z_depth_indices: [B, code_seq_len]

    Outputs:
        If z_depth_indices is given:
            loss, z_refined_logits, z_refined_feature
        Else:
            z_refined_logits, z_refined_feature

    Typical config:
        dim=1024
        z_rgb_feature_dim=4096
        codebook_size=8
        code_seq_len=4
    """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        code_seq_len=4,
        z_rgb_feature_dim=4096,
        z_rgb_feature_dropout=0.0,
        hidden_mult=1,
        num_mlp_layers=2,
        **unused_kwargs,
    ):
        """
        unused_kwargs allows this class to ignore old Stage25 args such as:
            image_size, patch_size, spatial_depth, dim_head, heads, channels,
            attn_dropout, ff_dropout

        This makes it easier to reuse old training scripts while switching to
        the Stage 2.5.2 no-depth model.
        """
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.code_seq_len = code_seq_len
        self.z_rgb_feature_dim = z_rgb_feature_dim

        hidden_dim = int(dim * hidden_mult)

        layers = [
            nn.LayerNorm(z_rgb_feature_dim),
            nn.Linear(z_rgb_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(z_rgb_feature_dropout),
        ]

        for _ in range(max(num_mlp_layers - 1, 0)):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(z_rgb_feature_dropout),
            ])

        layers.extend([
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        ])

        self.z_rgb_feature_proj = nn.Sequential(*layers)

        # Slot embeddings let the same refined feature produce different
        # predictions for each latent position.
        self.slot_embed = nn.Parameter(torch.randn(code_seq_len, dim) * 0.02)

        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Predict one distribution over codebook IDs for each latent slot.
        self.head = nn.Linear(dim, codebook_size)

    def load(self, path, strict=False):
        """
        Load a Stage 2.5.2 checkpoint.

        Use strict=True for checkpoints trained with this exact class.
        Use strict=False if partially initializing from another Stage25 checkpoint.
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

    def encode_z_rgb_features(self, z_rgb_features):
        """
        Convert precomputed LAPA RGB features into latent prior feature.

        Args:
            z_rgb_features: [B, z_rgb_feature_dim], e.g. [B, 4096]

        Returns:
            z_refined_feature: [B, D]
        """
        assert z_rgb_features.ndim == 2, (
            f"Expected [B, z_rgb_feature_dim], got {z_rgb_features.shape}"
        )
        assert z_rgb_features.shape[1] == self.z_rgb_feature_dim, (
            f"Expected z_rgb_feature_dim={self.z_rgb_feature_dim}, "
            f"got {z_rgb_features.shape[1]}"
        )

        z_rgb_features = z_rgb_features.float()
        z_refined_feature = self.z_rgb_feature_proj(z_rgb_features)  # [B, D]

        return z_refined_feature

    def forward(
        self,
        z_rgb_features,
        z_depth_indices=None,
        **unused_kwargs,
    ):
        """
        Stage 2.5.2 forward.

        Args:
            z_rgb_features:  [B, z_rgb_feature_dim], e.g. [B, 4096]
            z_depth_indices: [B, code_seq_len], optional

        Returns:
            If z_depth_indices is provided:
                loss, z_refined_logits, z_refined_feature
            Else:
                z_refined_logits, z_refined_feature
        """
        z_refined_feature = self.encode_z_rgb_features(z_rgb_features)  # [B, D]

        # Create one token per latent slot.
        z_refined_tokens = z_refined_feature[:, None, :] + self.slot_embed[None, :, :]
        z_refined_tokens = self.slot_mlp(z_refined_tokens)

        z_refined_logits = self.head(z_refined_tokens)  # [B, L, codebook_size]

        if z_depth_indices is not None:
            loss = F.cross_entropy(
                z_refined_logits.reshape(-1, z_refined_logits.shape[-1]),
                z_depth_indices.reshape(-1).long(),
            )
            return loss, z_refined_logits, z_refined_feature

        return z_refined_logits, z_refined_feature

    def predict_indices(self, z_rgb_features, **unused_kwargs):
        """
        Convenience function for inference.

        Returns:
            z_pred_indices: [B, code_seq_len]
        """
        z_refined_logits, _ = self.forward(
            z_rgb_features=z_rgb_features,
            z_depth_indices=None,
        )
        return z_refined_logits.argmax(dim=-1)


# Backward-compatible alias if using this file standalone.
# Prefer importing LatentActionQuantizationStage252 explicitly.
Stage252Model = LatentActionQuantizationStage252
