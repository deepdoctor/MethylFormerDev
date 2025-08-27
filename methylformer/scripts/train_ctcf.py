
import os
import sys
from contextlib import nullcontext
from torch.cuda.amp import autocast 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

root_dir = "/media/desk16/zhiwei/paper_code/MethylFormer/methylformer"
os.chdir(root_dir)
sys.path.append(root_dir)
from utils.data_ctcf import MethylationToCTCFDataset
from utils.utils_nn import *
from models.model import CTCFTransformer


# ---------------------- printing helper ----------------------
def rprint(*args, **kwargs):
    print(*args, **kwargs)


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------- main ----------------------
def main() -> None:
    args,_ = parse_and_resolve_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"[env] device={device} cuda={torch.cuda.is_available()}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # enable_sdp_flash_if_available()

    # dataset
    if not os.path.exists(args.data):
        rprint(f"[error] Training data file not found: {args.data}")
        return

    train_dataset = MethylationToCTCFDataset(
        json_path=args.data,
        max_len=args.max_len,
        step=args.step,
        sos_value=0.0,
        pad_value=0.0,
        normalize_targets=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    if args.val_data and os.path.exists(args.val_data):
        val_dataset = MethylationToCTCFDataset(
            json_path=args.val_data,
            max_len=args.max_len,
            step=args.step,
            sos_value=0.0,
            pad_value=0.0,
            normalize_targets=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        val_loader = None

    # model
    ctcf_model = CTCFTransformer(d_model=args.d_model, nhead=args.nhead,
                                num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                                max_len=args.max_len).to(device)
    # optimizer & scheduler
    params = list(ctcf_model.parameters()) 
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rprint(f"[optim] Using AdamW, base lr={args.lr}")

    steps_per_epoch = len(train_loader)
    scheduler, sched_mode = build_scheduler(opt, args, steps_per_epoch)

    use_autocast = args.precision in ("fp16", "bf16") and device.type == "cuda"
    scaler = GradScaler(enabled=(args.precision == "fp16"))

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- training loop ----------
    for ep in range(args.epochs):
        ctcf_model.train()

        loss_sum = 0.0
        mae_sum = 0.0
        tok_sum = 0.0
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
                mu, logvar = ctcf_model(methylation, positions, dec_in, positions, memory_key_padding_mask=pad_mask)

                # loss_map = ((mu - targets) ** 2 * inv_var + logvar).masked_fill(pad_mask, 0.0)
                loss_map = ((mu - targets) ** 2  ).masked_fill(pad_mask, 0.0)

                loss = loss_map.sum() / ((~pad_mask).float().sum().clamp_min(1.0))
                mae_map = (torch.abs(mu - targets)).masked_fill(pad_mask, 0.0)

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

            loss_sum += loss_map.sum().detach().item()
            mae_sum += mae_map.sum().detach().item()
            tok_sum += (~pad_mask).float().sum().detach().item()

            if it % 100 == 0:
                lr_cur = opt.param_groups[0]['lr']
                rprint(f"Epoch {ep+1} It {it}/{len(train_loader)} | lr={lr_cur:.6g} | loss={loss.item():.4f}")

        train_loss = loss_sum / max(tok_sum, 1.0)
        train_mae = mae_sum / max(tok_sum, 1.0)
        rprint(f"[epoch {ep+1}] train_loss={train_loss:.4f} train_mae={train_mae:.4f}")

        # periodic validation
        do_val = (val_loader is not None) and ((ep + 1) % max(1, args.val_every) == 0)
        val_loss = None
        val_mae = None
                # periodic validation
        if do_val:
            edmod = ctcf_model.module if isinstance(ctcf_model, DDP) else ctcf_model
            val_loss, val_mae = run_eval( edmod, val_loader, device, precision=args.precision)
            rprint(f"[val] Epoch {ep+1}: val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")


        if scheduler is not None and sched_mode in ("epoch", "plateau"):
            metric = train_loss
            if sched_mode == "plateau":
                scheduler.step(metric)
            else:
                scheduler.step()

        if (ep + 1) % args.save_every == 0:
            ckpt = {
                "epoch": ep + 1,
                "ctcf_model": ctcf_model.state_dict(),
                "optim": opt.state_dict(),
                "args": vars(args),
            }
            path = os.path.join(args.save_dir, f"ctcf_ep{ep+1}.pt")
            torch.save(ckpt, path)
            rprint(f"[ckpt] saved {path}")


if __name__ == "__main__":
    main()
