# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 2D ResNet Components ====================

class ResidualBlock2D(nn.Module):
    """2D Residual Block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out)


class SmallResNet2D(nn.Module):
    """2D ResNet for voltage maps (96×96 → embed_dim)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block1 = ResidualBlock2D(32, 64)
        self.block2 = ResidualBlock2D(64, 128, stride=2)
        self.block3 = ResidualBlock2D(128, 256, stride=2)
        self.block4 = ResidualBlock2D(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x))


# ==================== 1D ResNet Components ====================

class ResidualBlock1D(nn.Module):
    """1D Residual Block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """1D ResNet for QHI/THI sequences (2×151 → embed_dim)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.block1 = ResidualBlock1D(32, 64)
        self.block2 = ResidualBlock1D(64, 128, stride=2)
        self.block3 = ResidualBlock1D(128, 256, stride=2)
        self.block4 = ResidualBlock1D(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x))


# ==================== Transformer 1D ====================

class Transformer1D(nn.Module):
    """Transformer for QHI/THI sequences with self-attention."""

    def __init__(
        self,
        seq_len: int = 151,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        out_dim: int = 128,
    ):
        super().__init__()
        self.inp = nn.Linear(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(seq_len * d_model, out_dim)

    def forward(self, qhi: torch.Tensor, thi: torch.Tensor) -> torch.Tensor:
        x = torch.stack([qhi, thi], dim=-1)  # (B, seq_len, 2)
        x = self.inp(x)  # (B, seq_len, d_model)
        x = self.transformer(x)  # (B, seq_len, d_model)
        x = x.reshape(x.shape[0], -1)  # (B, seq_len * d_model)
        return self.fc(x)  # (B, out_dim)


# ==================== Point Features MLP ====================

class PointMLP(nn.Module):
    """MLP for scalar health indicators (15 → embed_dim)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== Functional Modality Attention Fusion ====================

class FunctionalModalityAttentionHead(nn.Module):
    """
    Attention-based fusion with learnable modality importance.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        self.attention_network = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 modalities
        )

        self.map_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
        )

        self.seq_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
        )

        self.pt_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
        )

        # SOH regression head with sigmoid constraint
        self.regression = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Constrain to [0, 1]
        )

    def forward(
        self,
        map_emb: torch.Tensor,
        seq_emb: torch.Tensor,
        pt_emb: torch.Tensor,
        return_attention: bool = False,
    ):
        cat = torch.cat([map_emb, seq_emb, pt_emb], dim=1)  # (B, 3*embed_dim)
        att_logits = self.attention_network(cat)  # (B, 3)
        att_scores = torch.softmax(att_logits, dim=1)  # (B, 3)

        map_proj = self.map_proj(map_emb)
        seq_proj = self.seq_proj(seq_emb)
        pt_proj = self.pt_proj(pt_emb)

        weighted_fused = (
            att_scores[:, 0:1] * map_proj
            + att_scores[:, 1:2] * seq_proj
            + att_scores[:, 2:3] * pt_proj
        )  # (B, embed_dim)

        pred = self.regression(weighted_fused).squeeze(1)  # (B,)

        if return_attention:
            return pred, att_scores
        return pred


# ==================== Fusion Models ====================

class FusionModelResNet(nn.Module):
    """Multi-modal SOH model with ResNet sequence processor."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.map_net = SmallResNet2D(out_dim=embed_dim)
        self.seq_net = ResNet1D(out_dim=embed_dim)
        self.pt_net = PointMLP(out_dim=embed_dim)
        self.fusion_head = FunctionalModalityAttentionHead(embed_dim)

    def forward(
        self,
        voltage_map: torch.Tensor,
        qhi: torch.Tensor,
        thi: torch.Tensor,
        scalar_features: torch.Tensor,
        return_attention: bool = False,
    ):
        if voltage_map.dim() == 3:
            voltage_map = voltage_map.unsqueeze(1)

        map_emb = self.map_net(voltage_map)
        seq_input = torch.stack([qhi, thi], dim=1)  # (B, 2, 151)
        seq_emb = self.seq_net(seq_input)
        pt_emb = self.pt_net(scalar_features)

        return self.fusion_head(map_emb, seq_emb, pt_emb, return_attention)


class FusionModelTransformer(nn.Module):
    """Multi-modal SOH model with Transformer sequence processor."""

    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 151,
    ):
        super().__init__()
        self.map_net = SmallResNet2D(out_dim=embed_dim)
        self.seq_net = Transformer1D(
            seq_len=seq_len,
            nhead=nhead,
            num_layers=num_layers,
            out_dim=embed_dim,
        )
        self.pt_net = PointMLP(out_dim=embed_dim)
        self.fusion_head = FunctionalModalityAttentionHead(embed_dim)

    def forward(
        self,
        voltage_map: torch.Tensor,
        qhi: torch.Tensor,
        thi: torch.Tensor,
        scalar_features: torch.Tensor,
        return_attention: bool = False,
    ):
        if voltage_map.dim() == 3:
            voltage_map = voltage_map.unsqueeze(1)

        map_emb = self.map_net(voltage_map)
        seq_emb = self.seq_net(qhi, thi)
        pt_emb = self.pt_net(scalar_features)

        return self.fusion_head(map_emb, seq_emb, pt_emb, return_attention)


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("architecture", "resnet")
    embed_dim = model_cfg.get("embed_dim", 128)

    if arch == "resnet":
        return FusionModelResNet(embed_dim=embed_dim)

    if arch == "transformer":
        nhead = model_cfg.get("nhead", 4)
        num_layers = model_cfg.get("num_layers", 2)
        seq_len = model_cfg.get("seq_len", 151)
        return FusionModelTransformer(
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=seq_len,
        )

    raise ValueError(f"Unknown architecture: {arch!r}. Use 'resnet' or 'transformer'.")


FusionModel = FusionModelResNet
