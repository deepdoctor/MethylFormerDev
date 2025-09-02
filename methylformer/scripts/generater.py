
"""
generate_ctcf.py
================
Standalone CTCF sequence generator:
- restore model from a training checkpoint
- iterate over a dataset and generate sequences in batches
- support mean (greedy) and sample (Gaussian) modes
"""
import os
import argparse
import json
from typing import Optional

import torch
from torch.utils.data import DataLoader
root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer"
os.chdir(root_dir)
import sys 
sys.path.append(root_dir)

from utils.data_ctcf import MethylationToCTCFDataset
from models.model import CTCFTransformer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the generator.

    Returns
    -------
    argparse.Namespace
        Contains dataset path, checkpoint, batch_size, precision, output path, etc.
    """
    p = argparse.ArgumentParser(description="Generate CTCF sequences from a trained model")
    p.add_argument("--data", type=str, default = "data/valid_ctcf.json",help="dataset json")
    p.add_argument("--ckpt", type=str, default = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer/checkpoints_aws/ctcf_ddp_ep50.pt",help="checkpoint path saved by train_ctcf_ddp.py")
    p.add_argument("--max_len", type=int, default=4000)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32","fp16","bf16"])
    p.add_argument("--mode", type=str, default="mean", choices=["mean","sample"])
    p.add_argument("--sigma_scale", type=float, default=1.0)
    p.add_argument("--out", type=str, default="predictions_ctcf.pt")
    p.add_argument("--config", type=str, default="", help="optional JSON config to read defaults (will not override CLI)")
    return p.parse_args()


def main() -> None:
    """
    Main:
      - enable SDPA
      - build dataset/loader
      - instantiate & load models
      - generate in mean/sample mode and save outputs
    """
    args = parse_args()

    # (Optional) read a config file as a reference; this script does not override CLI args.
    if args.config:
        with open(args.config, "r") as f:
            _ = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.data is None or args.ckpt is None:
        raise SystemExit("Please provide --data and --ckpt")

    ds = MethylationToCTCFDataset(
        json_path=args.data, max_len=args.max_len, step=args.step,
        sos_value=0.0, pad_value=0.0, normalize_targets=False
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=(args.num_workers>0))

    # instantiate models (must match training dims)
    ctcf_model = CTCFTransformer(d_model=128, nhead=4, num_layers=4, max_len=args.max_len)

    ckpt = torch.load(args.ckpt, map_location=device)
    ctcf_model.load_state_dict(ckpt["ctcf_model"])
    ctcf_model.to(device).eval();
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            methylation = batch["methylation"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            preds, mu_seq, logvar_seq = ctcf_model.generate(methylation,positions,
                 positions,start_tokens=None, max_len=args.max_len)
            # print(preds)
            preds_all.append(preds.float().cpu())
            print("generate")

    preds = torch.cat(preds_all, dim=0) if len(preds_all)>0 else torch.empty(0)
    torch.save({"preds": preds}, args.out)
    print(f"[gen] Saved predictions to {args.out}")


if __name__ == "__main__":
    main()

    # python scripts/generater.py --max_len 201 --data data/valid_merged_bin20.json --ckpt weights_gpu_bin20/ctcf_ep1.pt --mode mean --out preds_ctcf.pt

