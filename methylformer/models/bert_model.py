
import torch
import torch.nn as nn

class MethylationBERT(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_len=512):
        super(MethylationBERT, self).__init__()
        self.meth_embed = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Linear(d_model, 1)

    def forward(self, methylation, position_ids):
        # methylation, position_ids: [B, L]
        meth_emb = self.meth_embed(methylation.unsqueeze(-1))
        pos_emb = self.pos_embed(position_ids.unsqueeze(-1))
        x = meth_emb + pos_emb + self.position_encoding[:, :methylation.size(1), :]
        out = self.transformer(x)
        prediction = self.regressor(out).squeeze(-1)  # [B, L]
        return prediction
