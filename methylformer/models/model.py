import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

def causal_mask(sz, device=None):
    if device is None:
        device = torch.device("cpu")
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)
    return mask  # True = block

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class CTCFEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4,dim_feedforward=2048, max_len=512):
        super().__init__()
        self.enc_in_proj = nn.Linear(1, d_model)
        self.enc_pos_proj = nn.Linear(1, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, enc_in, enc_pos, src_key_padding_mask=None):
        B, L = enc_in.size()
        # x = self.enc_in_proj(enc_in.unsqueeze(-1)) + self.enc_pos_proj(enc_pos.unsqueeze(-1))
        x = self.enc_in_proj(enc_in.unsqueeze(-1))
        x = self.pos_encoding(x)  # 加入正余弦位置编码
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)
        return memory


class CTCFDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=2048,max_len=512):
        super().__init__()
        self.dec_in_proj = nn.Linear(1, d_model)
        self.dec_pos_proj = nn.Linear(1, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, 2)  # predict mu and logvar

    def forward(self, dec_in, dec_pos, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        B, L = dec_in.size()
        device = dec_in.device
        # x = self.dec_in_proj(dec_in.unsqueeze(-1)) + self.dec_pos_proj(dec_pos.unsqueeze(-1))
        x = self.dec_in_proj(dec_in.unsqueeze(-1)) 

        x = self.pos_encoding(x)  
        tgt_mask = causal_mask(L, device=device)

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = self.head(out)
        
        mu = F.relu(out[..., 0])
        logvar = out[..., 1]
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        return mu, logvar


class CTCFTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4,dim_feedforward=2048, max_len=512):
        super().__init__()
        self.encoder = CTCFEncoder(d_model, nhead, num_layers,dim_feedforward, max_len)
        self.decoder = CTCFDecoder(d_model, nhead, num_layers,dim_feedforward, max_len)
        self.max_len = max_len

    def forward(self, enc_in, enc_pos, dec_in, dec_pos,
                src_key_padding_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(enc_in, enc_pos, src_key_padding_mask=src_key_padding_mask)
        mu, logvar = self.decoder(dec_in, dec_pos, memory,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
        return mu, logvar

    @torch.no_grad()
    def generate(self,
                 enc_in,
                 enc_pos,
                 dec_pos_seq,
                 start_tokens=None,
                 max_len=None,
                 greedy=True,
                 sample=False,
                 temperature=1.0,
                 memory_key_padding_mask=None,
                 tgt_key_padding_mask=None):
        """
        Autoregressive generation.

        Args:
            enc_in: (B, L_enc) encoder input values (float)
            enc_pos: (B, L_enc) encoder positions (float)
            dec_pos_seq: (B, T_out) decoder positions (float) for each generation timestep.
                         The generate loop will produce up to T_out tokens.
            start_tokens: None or tensor (B, 1) initial decoder token(s). If None, uses zeros.
            max_len: int or None - maximum output length; if None uses dec_pos_seq.shape[1].
            greedy: if True, chooses deterministic prediction (mu) as next token.
            sample: if True, sample from predicted Gaussian N(mu, sigma^2). (sample and greedy are exclusive; sample wins)
            temperature: float scaling for sampling (applies to sigma and to mu if you want more stochasticity)
            memory_key_padding_mask, tgt_key_padding_mask: forwarded to decoder if needed

        Returns:
            generated: (B, T) tensor of generated values (float)
            mu_seq: (B, T) predicted mu for each timestep
            logvar_seq: (B, T) predicted logvar for each timestep
        """
        device = enc_in.device
        B = enc_in.size(0)
        T_out = dec_pos_seq.size(1)
        if max_len is None:
            max_len = T_out
        else:
            max_len = min(max_len, T_out)

        # prepare memory once
        memory = self.encoder(enc_in, enc_pos, src_key_padding_mask=memory_key_padding_mask)

        # initialize decoder input
        if start_tokens is None:
            # default start token = 0. shape (B,1)
            cur_dec = torch.zeros((B, 1), device=device, dtype=enc_in.dtype)
        else:
            # ensure shape (B,1)
            if start_tokens.dim() == 1:
                cur_dec = start_tokens.unsqueeze(1).to(device)
            else:
                cur_dec = start_tokens.to(device)

        # prepare pos sequence for decoder (will slice progressively)
        dec_pos_seq = dec_pos_seq.to(device)

        generated = []
        mu_all = []
        logvar_all = []

        for t in range(1, max_len + 1):
            # current decoder positions for steps [0..t-1]
            cur_pos = dec_pos_seq[:, :t]  # (B, t)

            # run decoder on current prefix
            mu, logvar = self.decoder(cur_dec, cur_pos, memory,
                                      memory_key_padding_mask=memory_key_padding_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask)  # (B, t)

            # take last timestep prediction
            mu_t = mu[:, -1]         # (B,)
            logvar_t = logvar[:, -1] # (B,)

            # decide next token
            if sample:
                std = (0.5 * logvar_t).exp() * float(temperature)
                eps = torch.randn_like(std)
                next_token = mu_t + eps * std
            else:
                # greedy deterministic: use mu
                next_token = mu_t if greedy else mu_t  # currently only greedy or sample supported

            # append
            generated.append(next_token)
            mu_all.append(mu_t)
            logvar_all.append(logvar_t)

            # append next_token to cur_dec for next iteration
            # cur_dec shape (B, t) -> expand to (B, t+1)
            cur_dec = torch.cat([cur_dec, next_token.unsqueeze(1)], dim=1)

        # stack along time
        generated = torch.stack(generated, dim=1)    # (B, T)
        mu_seq = torch.stack(mu_all, dim=1)          # (B, T)
        logvar_seq = torch.stack(logvar_all, dim=1)  # (B, T)

        return generated, mu_seq, logvar_seq


