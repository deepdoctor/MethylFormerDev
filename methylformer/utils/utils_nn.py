import torch
import math
import os
import sys
import time
import argparse
import json
from contextlib import nullcontext
from typing import Dict, Tuple, Optional
from torch.cuda.amp import autocast 
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn.functional as F
# ---------------------- Argparse & Config ----------------------
# ---------------------- printing helpers ----------------------
class RankPrinter:
    """Only prints from rank 0 unless force=True."""
    def __init__(self):
        self._is_main = True

    def set_is_main(self, is_main: bool):
        self._is_main = bool(is_main)

    def __call__(self, *args, force: bool = False, **kwargs):
        if self._is_main or force:
            print(*args, **kwargs)

rprint = RankPrinter()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def add_args(parser: argparse.ArgumentParser) -> None:
    # data
    parser.add_argument("--data", type=str, default="data/valid_ctcf.json", help="path to training dataset json")
    parser.add_argument("--val_data", type=str, default="data/valid_ctcf.json", help="path to validation dataset json (optional)")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    # scheduler
    parser.add_argument("--sched", type=str, default="cosine",
                        choices=["none", "cosine", "linear", "onecycle", "cosine_restart", "plateau"])
    parser.add_argument("--warmup_steps", type=int, default=-1, help="<=0 uses warmup_ratio")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--oc_div_factor", type=float, default=25.0)
    parser.add_argument("--oc_final_div_factor", type=float, default=1e4)
    parser.add_argument("--t0_steps", type=int, default=0, help="T_0 for cosine_restart in steps; 0 means 1 epoch worth")
    parser.add_argument("--t_mult", type=int, default=2)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=2)

    # validation
    parser.add_argument("--val_every", type=int, default=10, help="run validation every N epochs if val_data is provided")

    # architecture / tricks

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--compile", action="store_true", help="use torch.compile for model graphs")
    parser.add_argument("--grad_ckpt", action="store_true", help="enable gradient checkpointing (encoder stage)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)

    # generation
    parser.add_argument("--gen_only", action="store_true", help="Only run generation (no training)")
    parser.add_argument("--gen_ckpt", type=str, default="", help="Checkpoint path for generation; if empty, will use latest in save_dir")
    parser.add_argument("--gen_out", type=str, default="predictions_ctcf.pt", help="Path to save generated sequences")
    parser.add_argument("--gen_mode", type=str, default="mean", choices=["mean", "sample"], help="Use mu or sample from N(mu, sigma)")
    parser.add_argument("--gen_sigma_scale", type=float, default=1.0, help="scale for std when sampling")

    # config I/O
    parser.add_argument("--config", type=str, default="configs/train_ctcf_gpu_bin40.json", help="JSON config path to load")
    # parser.add_argument("--save_config", type=str, default="configs/train_ctcf_bin40.json", help="Save the resolved config to this JSON path and exit")
    parser.add_argument("--save_config", type=str, default="", help="Save the resolved config to this JSON path and exit")

    parser.add_argument("--dump_used_config", action="store_true", help="At training start, dump resolved config to save_dir/config_used.json")

    # NEW: distributed control
    parser.add_argument("--distributed", type=str, default="auto", choices=["auto", "ddp", "none"],
                        help="Force DDP on/off or auto-detect via WORLD_SIZE")
    parser.add_argument("--debug_print", action="store_true", help="Print extra runtime info (rank 0)")

def parse_and_resolve_args() -> Tuple[argparse.Namespace, bool]:
    parser = argparse.ArgumentParser(description="CTCF decoder training with DDP + accelerations + LR scheduler + validation + logging + config I/O")
    add_args(parser)

    defaults = parser.parse_args([])
    defaults_dict = vars(defaults).copy()

    cli = parser.parse_args()
    cli_dict = vars(cli).copy()

    merged = defaults_dict.copy()

    if cli_dict.get("config"):
        with open(cli_dict["config"], "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if k in merged:
                merged[k] = v

    for k, v in cli_dict.items():
        if v != defaults_dict.get(k):
            merged[k] = v

    if merged.get("save_config"):
        outp = merged["save_config"]
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        if is_main_process():
            rprint(f"[config] Wrote config to {outp}")
        return argparse.Namespace(**merged), True

    return argparse.Namespace(**merged), False


def build_scheduler(optimizer: torch.optim.Optimizer,
                    args: argparse.Namespace,
                    steps_per_epoch: int) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], Optional[str]]:
    total_steps = max(1, (steps_per_epoch * args.epochs + args.accum - 1) // args.accum)
    warmup_steps = args.warmup_steps
    if warmup_steps < 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    if args.sched == "none":
        return None, None

    if args.sched in ("cosine", "linear"):
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            if args.sched == "linear":
                return max(0.0, 1.0 - progress)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            if args.min_lr > 0:
                return (args.min_lr / args.lr) + (1.0 - (args.min_lr / args.lr)) * cos
            return cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, "step"

    if args.sched == "onecycle":
        pct_start = warmup_steps / max(1, total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=args.oc_div_factor,
            final_div_factor=args.oc_final_div_factor,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
        return scheduler, "step"

    if args.sched == "cosine_restart":
        T_0 = max(1, args.t0_steps if args.t0_steps > 0 else int(steps_per_epoch / max(1, args.accum)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=args.t_mult, eta_min=args.min_lr
        )
        return scheduler, "step"

    if args.sched == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
            verbose=is_main_process(),
        )
        return scheduler, "plateau"

    raise ValueError(f"Unknown scheduler {args.sched}")


# ---------------------- SDPA setup ----------------------
def enable_sdp_flash_if_available(verbose: bool = True) -> None:
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
        if verbose and is_main_process():
            rprint("[SDPA] Enabled flash + mem_efficient via torch.backends.cuda.sdp_kernel")
    except Exception:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            if verbose and is_main_process():
                rprint("[SDPA] Enabled flash + mem_efficient via legacy flags")
        except Exception as e:
            if verbose and is_main_process():
                rprint(f"[SDPA] Could not enable SDPA flash kernels: {e}")


@torch.no_grad()
def run_eval(model,
             loader: DataLoader,
             device: torch.device,
             precision: str = "bf16") -> Tuple[float, float]:
    model.eval()
    use_autocast = (precision in ("fp16", "bf16") and device.type == "cuda")
    dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

    loss_sum = torch.zeros(1, device=device)
    mae_sum = torch.zeros(1, device=device)
    tok_sum = torch.zeros(1, device=device)

    for batch in loader:
        methylation = batch["methylation"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        dec_in = batch.get("dec_in", torch.zeros_like(positions, dtype=methylation.dtype)).to(device, non_blocking=True)
        targets = batch.get("targets", None)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)

        ctx = autocast(dtype=dtype) if use_autocast else nullcontext()
        with ctx:
            mu, logvar = model(methylation, positions,dec_in, positions,memory_key_padding_mask=pad_mask)
            if targets is not None:
                targets = targets.to(device, non_blocking=True)
                inv_var = torch.exp(-logvar)
                loss_map = ((mu - targets) ** 2 ).masked_fill(pad_mask, 0.0)

                mae_map = (torch.abs(mu - targets)).masked_fill(pad_mask, 0.0)
                loss_sum += loss_map.sum()
                mae_sum += mae_map.sum()
                tok_sum += (~pad_mask).float().sum()

    loss_sum = all_reduce_sum(loss_sum)
    mae_sum = all_reduce_sum(mae_sum)
    tok_sum = all_reduce_sum(tok_sum)

    loss = (loss_sum / tok_sum.clamp_min(1.0)).item()
    mae = (mae_sum / tok_sum.clamp_min(1.0)).item()
    return loss, mae

def get_loss(mu, targets, pad_mask=None, loss_type="mse", logvar=None, delta=1.0):
    """
    Universal loss function selector with support for multiple loss types.

    Args:
        mu: Predicted values (tensor)
        targets: Ground truth values (tensor)
        pad_mask: Boolean mask for padding positions (True = ignore)
        loss_type: 
            - str: one of ["mse", "mae", "huber", "gaussian_nll", "wasserstein", "corr"]
            - dict: e.g. {"mse": 0.7, "corr": 0.3} for weighted combination
        logvar: Log-variance tensor, required if loss_type includes "gaussian_nll"
        delta: Smoothing parameter for Huber loss

    Returns:
        Scalar tensor loss
    """
    if pad_mask is None:
        pad_mask = torch.zeros_like(targets, dtype=torch.bool)

    def _single_loss(loss_name):
        denom = ((~pad_mask).float().sum().clamp_min(1.0))

        if loss_name == "mse":
            loss_map = ((mu - targets) ** 2).masked_fill(pad_mask, 0.0)
            return loss_map.sum() / denom

        elif loss_name == "mae":
            loss_map = (torch.abs(mu - targets)).masked_fill(pad_mask, 0.0)
            return loss_map.sum() / denom

        elif loss_name == "huber":
            loss_map = F.huber_loss(mu, targets, reduction="none", delta=delta).masked_fill(pad_mask, 0.0)
            return loss_map.sum() / denom

        elif loss_name == "gaussian_nll":
            assert logvar is not None, "logvar must be provided for gaussian_nll"
            inv_var = torch.exp(-logvar)
            loss_map = (((mu - targets) ** 2) * inv_var + logvar).masked_fill(pad_mask, 0.0)
            return loss_map.sum() / denom

        elif loss_name == "wasserstein":
            # 1D Wasserstein distance (approx via sorting)
            mu_sorted, _ = torch.sort(mu[~pad_mask].view(-1))
            tgt_sorted, _ = torch.sort(targets[~pad_mask].view(-1))
            return torch.mean(torch.abs(mu_sorted - tgt_sorted))

        elif loss_name == "corr":
            # Negative Pearson correlation
            x = mu[~pad_mask].view(-1)
            y = targets[~pad_mask].view(-1)
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
            return 1 - corr  # loss decreases when correlation increases

        else:
            raise ValueError(f"Unknown loss_type: {loss_name}")

    # Case 1: single loss (string)
    if isinstance(loss_type, str):
        return _single_loss(loss_type)

    # Case 2: multiple losses (dict with weights)
    elif isinstance(loss_type, dict):
        total_loss = 0.0
        for name, weight in loss_type.items():
            total_loss += weight * _single_loss(name)
        return total_loss

    else:
        raise TypeError("loss_type must be str or dict")
