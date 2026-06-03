from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LatentActionQuantizationStage25Model5(nn.Module):
    """
    Model 5 for LAPA-depth Stage 2.5.

    Pipeline:
        z_rgb_features -> z_depth_feature

    This version:
        - does NOT use depth1
        - does NOT use z_depth_indices
        - does NOT predict codebook IDs
        - trains only with continuous z_depth_feature ground truth

    Inputs:
        z_rgb_features: FloatTensor [B, z_rgb_feature_dim], usually [B, 4096]

    Target:
        z_depth_feature:
            FloatTensor [B, z_depth_feature_dim]
            or FloatTensor [B, code_seq_len, z_depth_feature_dim]
            if predict_token_features=True

    Outputs:
        If z_depth_feature is given:
            loss, logs, pred_z_depth_feature
        Else:
            pred_z_depth_feature

    Typical configs:
        If Stage 1 returns global encoder feature:
            z_depth_feature_dim=1024
            predict_token_features=False

        If Stage 1 returns pooled VQ feature:
            z_depth_feature_dim=32
            predict_token_features=False

        If Stage 1 returns token-level feature:
            z_depth_feature_dim=32 or 1024
            predict_token_features=True
    """

    def __init__(
        self,
        *,
        dim,
        code_seq_len=4,
        z_rgb_feature_dim=4096,
        z_rgb_feature_dropout=0.0,
        z_depth_feature_dim=1024,
        predict_token_features=False,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
        hidden_mult=2,
        num_mlp_layers=3,
        **unused_kwargs,
    ):
        """
        unused_kwargs lets older scripts pass unused args such as image_size,
        patch_size, spatial_depth, codebook_size, heads, etc. without breaking
        Model 5.
        """
        super().__init__()

        self.dim = int(dim)
        self.code_seq_len = int(code_seq_len)
        self.z_rgb_feature_dim = int(z_rgb_feature_dim)
        self.z_depth_feature_dim = int(z_depth_feature_dim)
        self.predict_token_features = bool(predict_token_features)

        self.feature_loss_weight = float(feature_loss_weight)
        self.cosine_loss_weight = float(cosine_loss_weight)

        hidden_dim = int(dim * hidden_mult)

        layers = [
            nn.LayerNorm(z_rgb_feature_dim),
            nn.Linear(z_rgb_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(z_rgb_feature_dropout),
        ]

        for _ in range(max(0, num_mlp_layers - 2)):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(z_rgb_feature_dropout),
            ])

        layers.extend([
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        ])

        self.encoder = nn.Sequential(*layers)

        # Used only if teacher z_depth_feature is token-level:
        # target shape [B, code_seq_len, z_depth_feature_dim].
        self.slot_embed = nn.Parameter(torch.randn(self.code_seq_len, dim) * 0.02)

        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.feature_head_global = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, z_depth_feature_dim),
        )

        self.feature_head_tokens = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, z_depth_feature_dim),
        )

    def load(self, path, strict=False):
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
        Args:
            z_rgb_features: [B, z_rgb_feature_dim], e.g. [B, 4096]

        Returns:
            encoded: [B, dim]
        """
        assert z_rgb_features.ndim == 2, (
            f"Expected [B, z_rgb_feature_dim], got {z_rgb_features.shape}"
        )
        assert z_rgb_features.shape[1] == self.z_rgb_feature_dim, (
            f"Expected z_rgb_feature_dim={self.z_rgb_feature_dim}, "
            f"got {z_rgb_features.shape[1]}"
        )

        return self.encoder(z_rgb_features.float())

    def predict(self, z_rgb_features):
        """
        Predict continuous z_depth_feature.

        Returns:
            pred_z_depth_feature:
                [B, z_depth_feature_dim] if predict_token_features=False
                [B, code_seq_len, z_depth_feature_dim] if predict_token_features=True
        """
        encoded = self.encode_z_rgb_features(z_rgb_features)

        if self.predict_token_features:
            slot_tokens = encoded[:, None, :] + self.slot_embed[None, :, :]
            slot_tokens = self.slot_mlp(slot_tokens)
            pred_z_depth_feature = self.feature_head_tokens(slot_tokens)
        else:
            pred_z_depth_feature = self.feature_head_global(encoded)

        return pred_z_depth_feature

    def compute_feature_loss(self, pred_z_depth_feature, gt_z_depth_feature):
        gt_z_depth_feature = gt_z_depth_feature.float()

        if pred_z_depth_feature.shape != gt_z_depth_feature.shape:
            raise RuntimeError(
                "pred_z_depth_feature and gt_z_depth_feature shape mismatch: "
                f"pred={tuple(pred_z_depth_feature.shape)}, "
                f"gt={tuple(gt_z_depth_feature.shape)}. "
                "Set z_depth_feature_dim and predict_token_features correctly."
            )

        mse_loss = F.mse_loss(pred_z_depth_feature, gt_z_depth_feature)

        pred_flat = pred_z_depth_feature.reshape(pred_z_depth_feature.shape[0], -1)
        gt_flat = gt_z_depth_feature.reshape(gt_z_depth_feature.shape[0], -1)
        cosine_loss = 1.0 - F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()

        return mse_loss, cosine_loss

    def forward(
        self,
        z_rgb_features,
        z_depth_feature=None,
    ):
        """
        Args:
            z_rgb_features:  [B, z_rgb_feature_dim]
            z_depth_feature: [B, D] or [B, L, D], optional

        Returns:
            If z_depth_feature is provided:
                loss, logs, pred_z_depth_feature
            Else:
                pred_z_depth_feature
        """
        pred_z_depth_feature = self.predict(z_rgb_features=z_rgb_features)

        if z_depth_feature is None:
            return pred_z_depth_feature

        mse_loss, cosine_loss = self.compute_feature_loss(
            pred_z_depth_feature=pred_z_depth_feature,
            gt_z_depth_feature=z_depth_feature,
        )

        loss = self.feature_loss_weight * mse_loss
        loss = loss + self.cosine_loss_weight * cosine_loss

        logs = {
            "loss": loss.detach(),
            "feature_mse_loss": mse_loss.detach(),
            "feature_cosine_loss": cosine_loss.detach(),
        }

        return loss, logs, pred_z_depth_feature

    def extract_z_depth_feature(self, z_rgb_features):
        return self.predict(z_rgb_features=z_rgb_features)
