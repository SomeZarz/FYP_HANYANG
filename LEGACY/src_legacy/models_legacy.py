import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 2D ResNet Components ====================

class ResidualBlock2D(nn.Module):
    """2D Residual Block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return F.relu(out)


class SmallResNet2D(nn.Module):
    """2D ResNet for voltage maps (96×96 → 128-dim embedding)."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = ResidualBlock2D(32, 64)
        self.block2 = ResidualBlock2D(64, 128, stride=2)
        self.block3 = ResidualBlock2D(128, 256, stride=2)
        self.block4 = ResidualBlock2D(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
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

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """1D ResNet for sequences (302-length → 128-dim embedding)."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock1D(32, 64)
        self.block2 = ResidualBlock1D(64, 128, stride=2)
        self.block3 = ResidualBlock1D(128, 256, stride=2)
        self.block4 = ResidualBlock1D(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
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
    """Transformer for sequences with self-attention."""

    def __init__(self, seq_len=151, d_model=64, nhead=4, num_layers=2, out_dim=128):
        super().__init__()
        self.inp = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(seq_len * d_model, out_dim)

    def forward(self, qhi, thi):
        x = torch.cat([qhi, thi], dim=1).permute(0, 2, 1)
        x = self.inp(x)
        x = self.transformer(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


# ==================== Point Features MLP ====================

class PointMLP(nn.Module):
    """MLP for scalar health indicators → 128-dim embedding."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ==================== Functional Modality Attention Fusion Head ====================

class FunctionalModalityAttentionHead(nn.Module):
    """
    Functional attention-based fusion with learnable modality importance weighting.

    This mechanism:
    1. Learns importance weights for each modality
    2. Projects embeddings to a common space
    3. Applies weighted combination of modality embeddings
    4. Predicts SOH from the fused representation

    Unlike pseudo-attention, these weights actually influence the prediction.
    """

    def __init__(self, embed_dim=128):
        super().__init__()

        # Attention network: learns modality importance weights
        self.attention_network = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 modalities: map, sequence, point
        )

        # Learnable projection matrices for each modality to common space
        self.map_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )
        self.pt_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )

        # SOH regression head: takes weighted fused features
        self.regression = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, map_emb, seq_emb, pt_emb, return_attention=False):
        """
        Args:
            map_emb: Voltage map embedding (batch, embed_dim)
            seq_emb: Sequence embedding (batch, embed_dim)
            pt_emb: Point features embedding (batch, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            pred: SOH predictions (batch,)
            att_scores: Attention weights if return_attention=True (batch, 3)
        """

        # Compute modality importance weights from concatenated features
        cat = torch.cat([map_emb, seq_emb, pt_emb], dim=1)  # (batch, 3*embed_dim)
        att_logits = self.attention_network(cat)  # (batch, 3)
        att_scores = torch.softmax(att_logits, dim=1)  # (batch, 3) - sums to 1 per sample

        # Project embeddings to common space
        map_proj = self.map_proj(map_emb)  # (batch, embed_dim)
        seq_proj = self.seq_proj(seq_emb)  # (batch, embed_dim)
        pt_proj = self.pt_proj(pt_emb)     # (batch, embed_dim)

        # Weighted combination: importance-weighted sum of modality embeddings
        weighted_fused = (
            att_scores[:, 0:1] * map_proj +  # Weight map modality: (batch, 1) * (batch, embed_dim)
            att_scores[:, 1:2] * seq_proj +  # Weight sequence modality
            att_scores[:, 2:3] * pt_proj     # Weight point modality
        )  # Shape: (batch, embed_dim)

        # Predict SOH from weighted fused features
        pred = self.regression(weighted_fused).squeeze(1)  # (batch,)

        if return_attention:
            return pred, att_scores
        return pred


# ==================== ResNet-based Fusion Model ====================

class FusionModelResNet(nn.Module):
    """Complete multi-modal SOH model with ResNet sequence processor."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.map_net = SmallResNet2D(out_dim=embed_dim)
        self.seq_net = ResNet1D(out_dim=embed_dim)
        self.pt_net = PointMLP(out_dim=embed_dim)
        self.fusion_head = FunctionalModalityAttentionHead(embed_dim)

    def forward(self, voltage_map, qhi, thi, scalar_features, return_attention=False):
        # Handle voltage map shape
        if voltage_map.dim() == 3:
            voltage_map = voltage_map.unsqueeze(1)

        # Process each modality
        map_emb = self.map_net(voltage_map)
        seq_emb = self.seq_net(torch.stack([qhi, thi], dim=1))
        pt_emb = self.pt_net(scalar_features)

        # Fuse and predict
        return self.fusion_head(map_emb, seq_emb, pt_emb, return_attention)


# ==================== Transformer-based Fusion Model ====================

class FusionModelTransformer(nn.Module):
    """Complete multi-modal SOH model with Transformer sequence processor."""

    def __init__(self, embed_dim=128, nhead=4, num_layers=2, seq_len=151):
        super().__init__()
        self.map_net = SmallResNet2D(out_dim=embed_dim)
        self.seq_net = Transformer1D(seq_len=seq_len, nhead=nhead, num_layers=num_layers, out_dim=embed_dim)
        self.pt_net = PointMLP(out_dim=embed_dim)
        self.fusion_head = FunctionalModalityAttentionHead(embed_dim)

    def forward(self, voltage_map, qhi, thi, scalar_features, return_attention=False):
        # Handle voltage map shape
        if voltage_map.dim() == 3:
            voltage_map = voltage_map.unsqueeze(1)

        # Process each modality
        map_emb = self.map_net(voltage_map)
        seq_emb = self.seq_net(qhi, thi)
        pt_emb = self.pt_net(scalar_features)

        # Fuse and predict
        return self.fusion_head(map_emb, seq_emb, pt_emb, return_attention)


# ==================== Model Factory ====================

def build_model(config):
    """
    Factory function to instantiate model based on configuration.

    Args:
        config: Configuration dictionary with model section containing:
            - architecture: "resnet" or "transformer"
            - embed_dim: Embedding dimension (default: 128)
            - nhead: Transformer heads (default: 4)
            - num_layers: Transformer layers (default: 2)
            - seq_len: Sequence length (default: 151)

    Returns:
        Instantiated model with unified forward signature.
    """
    model_config = config.get('model', {})
    arch = model_config.get('architecture', 'resnet')
    embed_dim = model_config.get('embed_dim', 128)

    if arch == 'resnet':
        return FusionModelResNet(embed_dim=embed_dim)
    elif arch == 'transformer':
        nhead = model_config.get('nhead', 4)
        num_layers = model_config.get('num_layers', 2)
        seq_len = model_config.get('seq_len', 151)
        return FusionModelTransformer(
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=seq_len
        )
    else:
        raise ValueError(f"Unknown architecture: '{arch}'. Choose 'resnet' or 'transformer'.")


# Backward compatibility alias
FusionModel = FusionModelResNet