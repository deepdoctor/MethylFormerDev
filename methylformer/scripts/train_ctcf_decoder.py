
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import os
root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformerfull"
os.chdir(root_dir)
import sys 
sys.path.append(root_dir)
from utils.data_ctcf import MethylationToCTCFDataset
from models.bert_model import MethylationBERT
from models.decoder import CTCFDecoder, EncoderWrapper

def gaussian_nll(mu, logvar, target, mask):
    var = torch.exp(logvar) + 1e-8
    nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
    nll = nll.masked_fill(mask, 0.0)
    denom = (~mask).float().sum().clamp_min(1.0)
    return nll.sum() / denom

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MethylationToCTCFDataset(
        json_path="data/train_ctcf.json",
        max_len=512,
        step=8,
        sos_value=0.0,
        pad_value=0.0,
        normalize_targets=False
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    enc = MethylationBERT(d_model=128, nhead=4, num_layers=4, max_len=512).to(device)
    ckpt_path = "checkpoints/methylation_bert.pt"
    if os.path.exists(ckpt_path):
        try:
            enc.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
            print("Loaded encoder checkpoint")
        except Exception as e:
            print("Could not load encoder checkpoint:", e)

    enc_wrap = EncoderWrapper(enc, d_model=128).to(device)
    dec = CTCFDecoder(d_model=128, nhead=4, num_layers=4, max_len=512).to(device)

    freeze_encoder = True
    if freeze_encoder:
        for p in enc_wrap.parameters():
            p.requires_grad = False

    params = list(dec.parameters()) + ([] if freeze_encoder else list(enc_wrap.parameters()))
    optimizer = AdamW(params, lr=3e-4, weight_decay=1e-2)

    epochs = 100
    for ep in range(epochs):
        dec.train()
        if not freeze_encoder:
            enc_wrap.train()
        tot_loss, tot_mae, steps = 0.0, 0.0, 0

        for batch in loader:
            methyl = batch["methylation"].to(device)
            pos = batch["positions"].to(device)
            dec_in = batch["dec_in"].to(device)
            y = batch["targets"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            memory = enc_wrap.encode(methyl, pos, key_padding_mask=pad_mask)
            mu, logvar = dec(dec_in, dec_pos=pos, memory=memory,
                             memory_key_padding_mask=pad_mask,
                             tgt_key_padding_mask=pad_mask)

            loss = gaussian_nll(mu, logvar, y, pad_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            with torch.no_grad():
                mae = (torch.abs(mu - y).masked_fill(pad_mask, 0.0).sum() /
                       (~pad_mask).float().sum().clamp_min(1.0))

            tot_loss += float(loss)
            tot_mae += float(mae)
            steps += 1

        print(f"Epoch {ep+1} | Loss {tot_loss/steps:.4f} | MAE {tot_mae/steps:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(dec.state_dict(), "checkpoints/ctcf_decoder.pt")
    print("Decoder checkpoint saved at checkpoints/ctcf_decoder.pt")

if __name__ == "__main__":
    main()
