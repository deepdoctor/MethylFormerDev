
import torch
import torch.nn as nn

def causal_mask(sz):
    # Build a causal mask for autoregressive decoding
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
    return mask  # True = block

class CTCFDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_len=512):
        super().__init__()
        self.dec_in_proj = nn.Linear(1, d_model)
        self.dec_pos_proj = nn.Linear(1, d_model)
        self.dec_pe = nn.Parameter(torch.randn(1, max_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, 2)  # predict mu and logvar

    def forward(self, dec_in, dec_pos, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        B, L = dec_in.size()
        x = self.dec_in_proj(dec_in.unsqueeze(-1)) + self.dec_pos_proj(dec_pos.unsqueeze(-1)) + self.dec_pe[:, :L, :]
        tgt_mask = causal_mask(L).to(x.device)

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.head(out)
        mu, logvar = out[..., 0], out[..., 1]
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        return mu, logvar

    @torch.no_grad()
    def generate(self, encoder, enc_in, enc_pos, max_len, start_token=0.0, memory_key_padding_mask=None):
        # Greedy decoding: predict step by step
        device = enc_in.device
        memory = encoder.encode(enc_in, enc_pos, memory_key_padding_mask)
        B = enc_in.size(0)
        dec_seq = torch.full((B, 1), float(start_token), dtype=torch.float32, device=device)
        dec_pos = torch.zeros_like(dec_seq)
        outputs = []
        for t in range(max_len):
            mu, logvar = self.forward(dec_seq, dec_pos, memory, memory_key_padding_mask, None)
            y_t = mu[:, -1]
            outputs.append(y_t.unsqueeze(1))
            dec_seq = torch.cat([dec_seq, y_t.unsqueeze(1)], dim=1)
            next_pos = (dec_pos[:, -1] + 1).unsqueeze(1)
            dec_pos = torch.cat([dec_pos, next_pos], dim=1)
        return torch.cat(outputs, dim=1)

class EncoderWrapper(nn.Module):
    # Wrapper to reuse encoder as memory provider
    def __init__(self, encoder, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(1, d_model)
        self.pos_proj = nn.Linear(1, d_model)

    def encode(self, methylation, positions, key_padding_mask=None):
        # Simple projection of inputs to d_model dimension
        x = self.proj(methylation.unsqueeze(-1)) + self.pos_proj(positions.unsqueeze(-1))
        return x
