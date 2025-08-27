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
from models.model import CTCFTransformer
from utils.utils_nn import *


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
def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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

#   "nhead": 4, 
#   "num_layers": 1, 
#   "dim_feedforward": 2048,

    # ---------- models ----------
    ctcf_model = CTCFTransformer(d_model=args.d_model, nhead=args.nhead,
                                num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                                max_len=args.max_len).to(device)
    if ddp:
        ctcf_model = DDP(ctcf_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ---------- optimizer & scheduler ----------
    fused_ok = hasattr(torch.optim, "AdamW") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    params = list(ctcf_model.parameters()) 
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

    # ---------- training loop ----------
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_csv = os.path.join(args.save_dir, "metrics.csv")
    csv_header_written = os.path.exists(metrics_csv) and os.path.getsize(metrics_csv) > 0

    try:
        for ep in range(args.epochs):
            if ddp:
                train_loader.sampler.set_epoch(ep)  # type: ignore[attr-defined]
                if val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
                    val_loader.sampler.set_epoch(ep)

            ctcf_model.train()

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

                    # decode
                    if isinstance(ctcf_model, DDP):
                        mu, logvar = ctcf_model.module(methylation, positions, dec_in, positions,memory_key_padding_mask=pad_mask)
                    else:
                        mu, logvar = ctcf_model(methylation, positions,dec_in, positions,memory_key_padding_mask=pad_mask)

                    inv_var = torch.exp(-logvar)
                    # loss_map = ((mu - targets) ** 2 * inv_var + logvar).masked_fill(pad_mask, 0.0)
                    loss_map = ((mu - targets) ** 2 ).masked_fill(pad_mask, 0.0)
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
                if is_main_process() and (it % 10 == 0):
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
                   # periodic validation
            if do_val:
                edmod = ctcf_model.module if isinstance(ctcf_model, DDP) else ctcf_model
                val_loss, val_mae = run_eval( edmod, val_loader, device, precision=args.precision)
                if is_main_process():
                    rprint(f"[val] Epoch {ep+1}: val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")


            # epoch/plateau scheduler step
            if scheduler is not None and sched_mode in ("epoch", "plateau"):
                metric = val_loss if (val_loss is not None) else train_loss
                if sched_mode == "plateau":
                    scheduler.step(metric)
                else:
                    scheduler.step()

            # checkpoint
            if is_main_process() and ((ep + 1) % args.save_every == 0):
                ckpt = {
                    "epoch": ep + 1,
                    "ctcf_model": (ctcf_model.module.state_dict() if isinstance(ctcf_model, DDP) else ctcf_model.state_dict()),
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
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

# ---------------------- Usage examples ----------------------
# 1) Single GPU (or CPU) plain python:
#    python scripts/train_ctcf_ddp.py --distributed none --config configs/train_ctcf_gpu.json
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
