"""
train_ctcf_ddp.py
===========================
Robust single-GPU and DDP training with:
- AMP (fp16/bf16)
- SDPA flash/memory‑efficient attention backends (when available)
- Gradient checkpointing (encoder)
- torch.compile (optional)
- Schedulers (cosine/linear/onecycle/cosine_restart/plateau) + warmup
- Periodic validation & per‑epoch JSONL/CSV logging
- JSON config I/O (load --config / export --save_config / dump used)
- CTCF sequence generation (mean or Gaussian sampling)
- **New**: sane DDP detection & logging, safer init, clearer prints, early sanity checks

Usage examples are shown at the bottom of this file.
"""

import os
import sys
import math
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
root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer"
os.chdir(root_dir)
import sys 
sys.path.append(root_dir)
from utils.data_ctcf import MethylationToCTCFDataset
from models.bert_model import MethylationBERT
from models.decoder import CTCFDecoder, EncoderWrapper


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


def _env(var: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(var)
    return v if v is not None else default


# ---------------------- distributed helpers ----------------------

def init_distributed(distributed_mode: str) -> Tuple[bool, int, torch.device, str]:
    """
    Initialize torch.distributed when launched via torchrun, with robust handling.

    Parameters
    ----------
    distributed_mode : {"auto","ddp","none"}

    Returns
    -------
    (ddp, local_rank, device, backend)
    """
    # Decide whether we *should* do DDP
    if distributed_mode == "none":
        ddp = False
    elif distributed_mode == "ddp":
        ddp = True
    else:  # auto
        world_size = int(_env("WORLD_SIZE", "1"))
        ddp = world_size > 1

    if not ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rprint.set_is_main(True)
        return False, local_rank, device, ""

    # DDP path
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Required env vars when using torchrun
    local_rank = int(_env("LOCAL_RANK", "0"))
    rank = int(_env("RANK", "0"))
    world_size = int(_env("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        # env:// is the default with torchrun; do not pass master_addr/port explicitly here
        dist.init_process_group(backend=backend, timeout=torch.distributed.constants.default_pg_timeout)

    # Set printer after dist is initialized
    rprint.set_is_main(dist.get_rank() == 0)
    rprint(f"[ddp] init: rank={dist.get_rank()} local_rank={local_rank} world_size={world_size} backend={backend}")

    return True, local_rank, device, backend


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


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


# ---------------------- schedulers ----------------------

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


# ---------------------- Argparse & Config ----------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    # data
    parser.add_argument("--data", type=str, default="data/train_ctcf.json", help="path to training dataset json")
    parser.add_argument("--val_data", type=str, default="", help="path to validation dataset json (optional)")
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
    parser.add_argument("--config", type=str, default="", help="JSON config path to load")
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


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------- loss ----------------------

def nll_gauss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    loss = ((mu - target) ** 2 * inv_var + logvar)
    loss = loss.masked_fill(mask, 0.0)
    denom = (~mask).float().sum().clamp_min(1.0)
    return loss.sum() / denom


# ---------------------- generation utilities ----------------------

@torch.no_grad()
def generate_ctcf(enc_wrap: EncoderWrapper,
                  dec: CTCFDecoder,
                  methylation: torch.Tensor,
                  positions: torch.Tensor,
                  pad_mask: torch.Tensor,
                  mode: str = "mean",
                  sigma_scale: float = 1.0,
                  precision: str = "bf16") -> torch.Tensor:
    device = positions.device
    use_autocast = (precision in ("fp16", "bf16") and device.type == "cuda")
    dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

    ctx = autocast(dtype=dtype) if use_autocast else nullcontext()
    with ctx:
        memory = enc_wrap.encode(methylation, positions, key_padding_mask=pad_mask)

        B, L = positions.shape
        dec_in = torch.zeros((B, L), device=device, dtype=memory.dtype)
        preds = torch.zeros((B, L), device=device, dtype=memory.dtype)

        valid_lens = (~pad_mask).sum(dim=1)
        max_len = int(valid_lens.max().item()) if valid_lens.numel() > 0 else L
        max_len = max(1, min(max_len, L))

        for t in range(max_len):
            mu, logvar = dec(dec_in, positions, memory, memory_key_padding_mask=pad_mask)
            mu_t = mu[:, t]
            if mode == "mean":
                next_val = mu_t
            else:
                std_t = (logvar[:, t] * 0.5).exp() * sigma_scale
                eps = torch.randn_like(std_t)
                next_val = mu_t + std_t * eps

            preds[:, t] = next_val
            if t + 1 < L:
                dec_in[:, t + 1] = next_val

        preds = preds.masked_fill(pad_mask, 0.0)
        return preds


# ---------------------- evaluation ----------------------

@torch.no_grad()
def run_eval(enc_wrap: EncoderWrapper,
             dec: CTCFDecoder,
             loader: DataLoader,
             device: torch.device,
             precision: str = "bf16") -> Tuple[float, float]:
    enc_wrap.eval()
    dec.eval()
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
            memory = enc_wrap.encode(methylation, positions, key_padding_mask=pad_mask)
            mu, logvar = dec(dec_in, positions, memory, memory_key_padding_mask=pad_mask)
            if targets is not None:
                targets = targets.to(device, non_blocking=True)
                inv_var = torch.exp(-logvar)
                loss_map = ((mu - targets) ** 2 * inv_var + logvar).masked_fill(pad_mask, 0.0)
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


# ---------------------- main ----------------------

def main() -> None:
    # Unbuffered stdout so early prints actually appear even if output is piped
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Parse/config
    args, only_save_cfg = parse_and_resolve_args()
    if only_save_cfg:
        return

    # Reproducibility
    set_seed(args.seed)

    # DDP or single process
    ddp, local_rank, device, backend = init_distributed(args.distributed)

    # Make sure rank0 prints only
    rprint.set_is_main(is_main_process())

    # Early environment banner
    rprint(
        f"[env] ddp={'on' if ddp else 'off'} backend={backend or '—'} device={device} "
        f"cuda={torch.cuda.is_available()} bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

    # torch._dynamo.config.suppress_errors = True
    # Performance knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    enable_sdp_flash_if_available()

    # ---------- sanity checks ----------
    if not os.path.exists(args.data):
        rprint(f"[error] Training data file not found: {args.data}", force=True)
        return
    if args.val_data and (not os.path.exists(args.val_data)):
        rprint(f"[warn] Validation data file not found, disabling val: {args.val_data}")
        args.val_data = ""

    # ---------- datasets & loaders ----------
    train_dataset = MethylationToCTCFDataset(
        json_path=args.data,
        max_len=args.max_len,
        step=args.step,
        sos_value=0.0,
        pad_value=0.0,
        normalize_targets=False,
    )
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    if args.val_data:
        val_dataset = MethylationToCTCFDataset(
            json_path=args.val_data,
            max_len=args.max_len,
            step=args.step,
            sos_value=0.0,
            pad_value=0.0,
            normalize_targets=False,
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ddp else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        val_loader = None

    # ---------- models ----------
    enc = MethylationBERT(d_model=128, nhead=8, num_layers=4, max_len=args.max_len)
    dec = CTCFDecoder(d_model=128, nhead=8, num_layers=4, max_len=args.max_len)
    enc_wrap = EncoderWrapper(enc, d_model=128)

    if args.compile:
        try:
            enc = torch.compile(enc)
            dec = torch.compile(dec)
            enc_wrap = torch.compile(enc_wrap)
            rprint("[compile] torch.compile enabled")
        except Exception as e:
            rprint(f"[compile] torch.compile failed: {e}")

    enc.to(device); dec.to(device); enc_wrap.to(device)

    if ddp:
        enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        # Decoder often has conditional branches; keep unused detection on to avoid hangs
        dec = DDP(dec, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        enc_wrap = DDP(enc_wrap, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ---------- optimizer & scheduler ----------
    fused_ok = hasattr(torch.optim, "AdamW") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    params = list(enc.parameters()) + list(dec.parameters()) + list(enc_wrap.parameters())
    opt = (torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=fused_ok)
           if fused_ok else torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay))
    rprint(f"[optim] Using {'fused ' if fused_ok else ''}AdamW, base lr={args.lr}")

    steps_per_epoch = len(train_loader)
    scheduler, sched_mode = build_scheduler(opt, args, steps_per_epoch)
    if scheduler is None:
        rprint("[sched] No LR scheduler")
    else:
        rprint(f"[sched] Using {args.sched} scheduler, step_mode={sched_mode}, warmup_steps={args.warmup_steps}({args.warmup_ratio} ratio)")

    use_autocast = args.precision in ("fp16", "bf16") and device.type == "cuda"
    scaler = GradScaler(enabled=(args.precision == "fp16"))

    # dump resolved config for reproducibility
    if is_main_process() and args.dump_used_config:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "config_used.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        rprint(f"[config] Saved resolved config to {os.path.join(args.save_dir, 'config_used.json')}")

    # ---------- generation-only mode ----------
    if args.gen_only:
        ckpt_path = args.gen_ckpt
        if ckpt_path == "":
            os.makedirs(args.save_dir, exist_ok=True)
            cands = [os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir) if f.endswith(".pt")]
            ckpt_path = sorted(cands)[-1] if cands else ""
        if ckpt_path == "":
            rprint("[gen] No checkpoint found. Provide --gen_ckpt or ensure checkpoints exist in save_dir.", force=True)
            return
        rprint(f"[gen] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        (enc.module if isinstance(enc, DDP) else enc).load_state_dict(ckpt["enc"])
        (dec.module if isinstance(dec, DDP) else dec).load_state_dict(ckpt["dec"])
        (enc_wrap.module if isinstance(enc_wrap, DDP) else enc_wrap).load_state_dict(ckpt["enc_wrap"])

        enc.eval(); dec.eval(); enc_wrap.eval()
        all_preds = []
        with torch.no_grad():
            for batch in train_loader:
                methylation = batch["methylation"].to(device, non_blocking=True)
                positions = batch["positions"].to(device, non_blocking=True)
                pad_mask = batch["pad_mask"].to(device, non_blocking=True)

                ewrap = enc_wrap.module if isinstance(enc_wrap, DDP) else enc_wrap
                dmod = dec.module if isinstance(dec, DDP) else dec
                preds = generate_ctcf(ewrap, dmod, methylation, positions, pad_mask,
                                      mode=args.gen_mode, sigma_scale=args.gen_sigma_scale, precision=args.precision)
                all_preds.append(preds.float().cpu())

        out_path = args.gen_out
        if ddp:
            root, ext = os.path.splitext(out_path)
            out_path = f"{root}.rank{local_rank}{ext}"
        preds_cat = torch.cat(all_preds, dim=0) if len(all_preds) > 0 else torch.empty(0)
        torch.save({"preds": preds_cat}, out_path)
        rprint(f"[gen] Saved predictions to {out_path}")
        barrier()
        return

    # ---------- training loop ----------
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    metrics_csv = os.path.join(args.save_dir, "metrics.csv")
    csv_header_written = os.path.exists(metrics_csv) and os.path.getsize(metrics_csv) > 0

    start = time.time()

    try:
        for ep in range(args.epochs):
            if ddp:
                train_loader.sampler.set_epoch(ep)  # type: ignore[attr-defined]
                if val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
                    val_loader.sampler.set_epoch(ep)

            enc.train(); dec.train(); enc_wrap.train()

            # token-weighted accumulators
            loss_sum = torch.zeros(1, device=device)
            mae_sum = torch.zeros(1, device=device)
            tok_sum = torch.zeros(1, device=device)

            steps = 0
            opt.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                methylation = batch["methylation"].to(device, non_blocking=True)
                positions = batch["positions"].to(device, non_blocking=True)
                dec_in = batch["dec_in"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                pad_mask = batch["pad_mask"].to(device, non_blocking=True)

                ctx = autocast(dtype=(torch.bfloat16 if args.precision == "bf16" else torch.float16)) if use_autocast else nullcontext()
                with ctx:
                    # encode
                    if args.grad_ckpt:
                        def _enc_func(m, p, km):
                            return (enc_wrap.module.encode(m, p, key_padding_mask=km) if isinstance(enc_wrap, DDP)
                                    else enc_wrap.encode(m, p, key_padding_mask=km))
                        memory = checkpoint(_enc_func, methylation, positions, pad_mask, use_reentrant=False)
                    else:
                        memory = (enc_wrap.module.encode(methylation, positions, key_padding_mask=pad_mask) if isinstance(enc_wrap, DDP)
                                  else enc_wrap.encode(methylation, positions, key_padding_mask=pad_mask))
                    # decode
                    if isinstance(dec, DDP):
                        mu, logvar = dec.module(dec_in, positions, memory, memory_key_padding_mask=pad_mask)
                    else:
                        mu, logvar = dec(dec_in, positions, memory, memory_key_padding_mask=pad_mask)

                    inv_var = torch.exp(-logvar)
                    loss_map = ((mu - targets) ** 2 * inv_var + logvar).masked_fill(pad_mask, 0.0)
                    loss = loss_map.sum() / ((~pad_mask).float().sum().clamp_min(1.0))
                    mae_map = (torch.abs(mu - targets)).masked_fill(pad_mask, 0.0)

                # backward + step (with accumulation)
                if args.precision == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                steps += 1
                if steps % args.accum == 0:
                    if args.precision == "fp16":
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    if scheduler is not None and sched_mode in ("step",):
                        scheduler.step()

                # accumulate (token-weighted)
                loss_sum += loss_map.sum().detach()
                mae_sum += mae_map.sum().detach()
                tok_sum += (~pad_mask).float().sum().detach()

                # occasional log
                if is_main_process() and (it % 100 == 0):
                    lr_cur = opt.param_groups[0]['lr']
                    rprint(f"Epoch {ep+1} It {it}/{len(train_loader)} | lr={lr_cur:.6g} | loss={loss.item():.4f}")

            # reduce training metrics across ranks
            loss_sum = all_reduce_sum(loss_sum)
            mae_sum = all_reduce_sum(mae_sum)
            tok_sum = all_reduce_sum(tok_sum)
            train_loss = (loss_sum / tok_sum.clamp_min(1.0)).item()
            train_mae = (mae_sum / tok_sum.clamp_min(1.0)).item()

            # periodic validation
            do_val = (val_loader is not None) and ((ep + 1) % max(1, args.val_every) == 0)
            val_loss = None
            val_mae = None
            if do_val:
                ewrap = enc_wrap.module if isinstance(enc_wrap, DDP) else enc_wrap
                dmod = dec.module if isinstance(dec, DDP) else dec
                val_loss, val_mae = run_eval(ewrap, dmod, val_loader, device, precision=args.precision)
                if is_main_process():
                    rprint(f"[val] Epoch {ep+1}: val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

            # epoch/plateau scheduler step
            if scheduler is not None and sched_mode in ("epoch", "plateau"):
                metric = val_loss if (val_loss is not None) else train_loss
                if sched_mode == "plateau":
                    scheduler.step(metric)
                else:
                    scheduler.step()

            # logging
            if is_main_process():
                lr_now = opt.param_groups[0]['lr']
                rprint(f"Epoch {ep+1}/{args.epochs} | train_loss {train_loss:.4f} | train_mae {train_mae:.4f} | time {time.time()-start:.1f}s | lr={lr_now:.6g}")
                # JSONL
                rec = {
                    "epoch": ep + 1,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "lr": lr_now,
                    "time_sec": time.time() - start,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                # CSV
                import csv
                write_header = not csv_header_written and (not os.path.exists(metrics_csv) or os.path.getsize(metrics_csv) == 0)
                with open(metrics_csv, "a", newline="") as fcsv:
                    writer = csv.writer(fcsv)
                    if write_header:
                        writer.writerow(["epoch", "train_loss", "train_mae", "val_loss", "val_mae", "lr", "time_sec"])
                    writer.writerow([ep + 1, f"{train_loss:.6f}", f"{train_mae:.6f}",
                                     (f"{val_loss:.6f}" if val_loss is not None else ""),
                                     (f"{val_mae:.6f}" if val_mae is not None else ""),
                                     f"{lr_now:.8f}", f"{time.time()-start:.3f}"])
                csv_header_written = True

            # checkpoint
            if is_main_process() and ((ep + 1) % args.save_every == 0):
                ckpt = {
                    "epoch": ep + 1,
                    "enc": (enc.module.state_dict() if isinstance(enc, DDP) else enc.state_dict()),
                    "dec": (dec.module.state_dict() if isinstance(dec, DDP) else dec.state_dict()),
                    "enc_wrap": (enc_wrap.module.state_dict() if isinstance(enc_wrap, DDP) else enc_wrap.state_dict()),
                    "optim": opt.state_dict(),
                    "args": vars(args),
                }
                path = os.path.join(args.save_dir, f"ctcf_ddp_ep{ep+1}.pt")
                torch.save(ckpt, path)
                rprint(f"[ckpt] saved {path}")
    except KeyboardInterrupt:
        rprint("[train] Interrupted by user, attempting clean shutdown…", force=True)
    finally:
        barrier()
        if is_main_process():
            rprint("Training complete.")


if __name__ == "__main__":
    main()

# ---------------------- Usage examples ----------------------
# 1) Single GPU (or CPU) plain python:
#    python scripts/train_ctcf_ddp.py --distributed none --config configs/train_ctcf_single_gpu.json
#
# 2) Auto-detect (WORLD_SIZE>1 => DDP, else single):
#    python scripts/train_ctcf_ddp.py --distributed auto --config your.json
#
# 3) Single machine, 4 GPUs (DDP):
#    torchrun --nproc_per_node=4 scripts/train_ctcf_ddp.py \
#        --distributed ddp --config configs/train_ctcf_multi_gpu.json
#
# 4) Generation only (on rank0 prints):
#    torchrun --nproc_per_node=4 scripts/train_ctcf_ddp.py --distributed ddp \
#        --gen_only --gen_ckpt checkpoints/ctcf_ddp_ep10.pt --gen_mode sample --gen_sigma_scale 1.0
