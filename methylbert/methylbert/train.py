
import argparse
import os
from typing import Dict
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylbert/methylbert"
root_dir = "/groups/adv2105_gp/yichen/deep/MethylFormerDev/methylbert/methylbert"

os.chdir(root_dir)
sys.path.append(root_dir)
from utils import set_seed, load_config, WarmupCosineLR
from data import build_dataloader
from model import MaskedValueBERT, masked_regression_loss

def parse_overrides(override_list):
    # Parse "a.b.c=123" CLI overrides
    overrides = {}
    for item in override_list or []:
        key, value = item.split("=", 1)
        # try int/float/bool/json parsing
        if value.lower() in ("true", "false"):
            v = value.lower() == "true"
        else:
            try:
                if "." in value:
                    v = float(value)
                    if v.is_integer():
                        v = int(v)
                else:
                    v = int(value)
            except ValueError:
                v = value
        overrides[key] = v
    return overrides

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/media/desk16/zhiwei/paper_code/MethylFormer/methylbert/methylbert/configs/default.yaml", help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="Key=Value overrides (e.g., training.batch_size=32)")
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    cfg = load_config(args.config, overrides)

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders (train-only for pretraining; split outside if needed)
    train_loader = build_dataloader(
        json_path=cfg["data"]["json_path"],
        cfg_data=cfg["data"],
        cfg_masking=cfg["masking"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        prefetch_factor=cfg["data"]["prefetch_factor"]
    )

    # Model
    model = MaskedValueBERT(**cfg["model"]).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=tuple(cfg["training"]["betas"]),
        eps=cfg["training"]["eps"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    total_steps = cfg["training"]["max_steps"]
    scheduler = WarmupCosineLR(optimizer, cfg["training"]["warmup_steps"], total_steps)

    scaler = GradScaler(enabled=cfg["training"]["amp"])

    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
    step = 0
    best_loss = float("inf")

    model.train()

    ckpt = torch.load("checkpoints/epoch1.pt", map_location=device)
    model.load_state_dict(ckpt["model"])


    
    for epoch in range(cfg["training"]["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        accum = cfg["training"]["gradient_accumulation"]
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            methyl = batch["methylation"].to(device)     # [B, L+1]
            pos = batch["positions"].to(device)          # [B, L+1]
            special = batch["special_ids"].to(device)    # [B, L+1]
            targets = batch["targets"].to(device)        # [B, L+1]
            loss_mask = batch["loss_mask"].to(device)    # [B, L+1]

            with autocast(enabled=cfg["training"]["amp"]):
                preds, _ = model(methyl, special)   # preds: [B, L+1]
                loss = masked_regression_loss(
                    preds, targets, loss_mask,
                    loss_type=cfg["loss"]["type"],
                    delta=cfg["loss"]["delta"]
                )
                loss = loss / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1
            pbar.set_postfix({"loss": float(loss.item() * accum), "lr": scheduler.get_last_lr()[0]})

            if 0 < total_steps <= step:
                break

        # Save checkpoint per epoch
        ckpt_path = os.path.join(cfg["training"]["save_dir"], f"epoch{epoch+1}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "epoch": epoch+1,
            "step": step
        }, ckpt_path)

        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(cfg["training"]["save_dir"], "best.pt"))

        if 0 < total_steps <= step:
            break

if __name__ == "__main__":
    main()
