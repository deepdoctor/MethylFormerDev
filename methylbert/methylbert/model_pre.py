
from typing import Optional
import torch.nn.functional as F
import math
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不可训练

    def forward(self, x):
        """
        x: [B, L, d_model]
        返回位置编码与输入同尺寸
        """
        return self.pe[:, :x.size(1), :]


ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

class PositionalProjector(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model)
        )
    def forward(self, pos):  # pos: [B, L]
        return self.net(pos.unsqueeze(-1))

class MaskedValueBERT(nn.Module):
    """
    Continuous-input BERT:
    - Input embedding = Linear(value) + (optional) PositionalProjector(positions)
    - Special tokens ([CLS], [MASK], [PAD]) have learned embeddings that *override* the value-based embedding.
    - Transformer encoder
    - Regression head outputs a scalar per position
    """
    def __init__(self, d_model=192, n_heads=6, n_layers=6, d_ff=384, dropout=0.1, pos_proj=True, pos_mlp_hidden=32, activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.pos_proj = PositionalProjector(d_model, pos_mlp_hidden) if pos_proj else None

        # Special token embeddings: indices 1=[CLS], 2=[MASK], 3=[PAD]
        self.special_embed = nn.Embedding(4, d_model)  # 0=none,1=CLS,2=MASK,3=PAD

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation=activation, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, methylation, positions, special_ids, attn_mask=None):
        # methylation: [B, L], positions: [B, L], special_ids: [B, L]
        val_emb = self.value_proj(methylation.unsqueeze(-1))  # [B, L, d]
        # if self.pos_proj is not None:
        #     val_emb = val_emb + self.pos_proj(positions)       # [B, L, d]

        # override with special embeddings where needed (special_ids > 0)
        special_mask = special_ids > 0
        if special_mask.any():
            val_emb[special_mask] = self.special_embed(special_ids[special_mask])

        x = self.dropout(val_emb)

        # Build attention padding mask: PAD tokens should be masked out
        pad_mask = special_ids.eq(3)  # [B, L] True for PAD
        enc = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, d]
        out = self.head(enc).squeeze(-1)  # [B, L]

        return out, enc  # return hidden states for potential CLS usage

def masked_regression_loss(pred, targets, loss_mask, loss_type="huber", delta=1.0):
    # Only compute on positions where loss_mask == 1
    if loss_type == "huber":
        return F.huber_loss(pred * loss_mask, targets * loss_mask, delta=delta, reduction="sum") / (loss_mask.sum().clamp(min=1.0))
    elif loss_type == "mse":
        return F.mse_loss(pred * loss_mask, targets * loss_mask, reduction="sum") / (loss_mask.sum().clamp(min=1.0))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
