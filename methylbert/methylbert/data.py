
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader
import orjson

class MethylationToCTCFDataset(Dataset):
    def __init__(self, json_path, max_len=512, step=1, sos_value=0.0, pad_value=0.0, normalize_targets=True):
        with open(json_path, "rb") as f:
            self.data = orjson.loads(f.read())
        self.max_len = max_len
        self.step = step
        self.sos_value = sos_value
        self.pad_value = pad_value
        self.normalize_targets = normalize_targets

    def _build_positions(self, item, seq_len):
        if "positions" in item and isinstance(item["positions"], list):
            pos = item["positions"][:seq_len]
        else:
            start = int(item["position"]) if "position" in item else 0
            pos = list(range(start, start + seq_len * self.step, self.step))
        return pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        it = self.data[idx]

        seq = it["methyValue"][: self.max_len]
        tgt = it.get("ctcfValue", [0.0]*len(seq))[: self.max_len]

        positions = self._build_positions(it, len(seq))

        L = len(seq)
        if L < self.max_len:
            pad_n = self.max_len - L
            seq = seq + [0.0] * pad_n
            tgt = tgt + [0.0] * pad_n
            positions = positions + [positions[-1] + self.step * i for i in range(1, pad_n + 1)]

        y = torch.tensor(tgt, dtype=torch.float32)
        dec_in = torch.tensor([self.sos_value] + tgt[:-1], dtype=torch.float32)

        valid_len = min(len(it.get("ctcfValue", seq)), self.max_len)
        pad_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if valid_len < self.max_len:
            pad_mask[valid_len:] = True

        return {
            "methylation": torch.tensor(seq, dtype=torch.float32),
            "positions": torch.tensor(positions, dtype=torch.float32),
            "dec_in": dec_in,
            "targets": y,
            "pad_mask": pad_mask
        }

class MaskedValueCollator:
    """
    BERT-style masking for continuous sequences.
    - Adds [CLS] at position 0 (learned embedding in the model).
    - 15% tokens selected; of those, 80% -> [MASK], 10% -> random value, 10% -> keep.
    - Loss computed only on selected (masked) positions (excluding CLS & PAD).
    """
    CLS_ID = 1
    MASK_ID = 2
    PAD_ID = 3

    def __init__(self, mlm_prob=0.15, mask_prob=0.8, random_prob=0.1, keep_prob=0.1, random_value_range=(0.0, 1.0)):
        self.mlm_prob = mlm_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.random_value_range = random_value_range

    def __call__(self, batch: List[Dict]):
        methyl = torch.stack([b["methylation"] for b in batch], dim=0)  # [B, L]
        pos = torch.stack([b["positions"] for b in batch], dim=0)       # [B, L]
        pad_mask = torch.stack([b["pad_mask"] for b in batch], dim=0)   # [B, L]

        B, L = methyl.shape

        # prepend CLS placeholder to positions/methyl; model will substitute CLS embedding
        cls_pad = torch.zeros(B, 1, dtype=methyl.dtype, device=methyl.device)
        methyl = torch.cat([cls_pad, methyl], dim=1)  # [B, L+1]
        pos = torch.cat([pos[:, :1] - 1, pos], dim=1) # simple pos for CLS: previous coordinate

        # extend pad_mask to account for CLS (which is never pad)
        pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device), pad_mask], dim=1)

        # Build special token id map per position
        special = torch.zeros(B, L+1, dtype=torch.long)  # 0=none; 1=CLS; 2=MASK; 3=PAD
        special[:, 0] = self.CLS_ID
        special[pad_mask] = self.PAD_ID

        # Choose candidates for masking: valid (not PAD, not CLS)
        candidate_mask = ~pad_mask
        candidate_mask[:, 0] = False

        # Select positions to mask
        rand = torch.rand(B, L+1)
        to_mask = (rand < self.mlm_prob) & candidate_mask

        # Prepare targets = original methyl values; we won't regress CLS/PAD
        targets = methyl.clone()

        # Apply 80/10/10 policy on the inputs
        policy_rand = torch.rand(B, L+1)
        mask_positions = (policy_rand < self.mask_prob) & to_mask
        random_positions = (policy_rand >= self.mask_prob) & (policy_rand < self.mask_prob + self.random_prob) & to_mask
        keep_positions = to_mask & ~(mask_positions | random_positions)

        # mark special MASK
        special[mask_positions] = self.MASK_ID

        # replace random positions with uniform random scalar
        if random_positions.any():
            low, high = self.random_value_range
            methyl[random_positions] = torch.empty_like(methyl[random_positions]).uniform_(low, high)

        # keep_positions: leave methyl as-is

        # For MASK positions, set methyl value to 0; model will replace by learned [MASK] embedding
        methyl[mask_positions] = 0.0

        # Build loss mask: only positions selected for masking (to_mask)
        loss_mask = to_mask.float()

        return {
            "methylation": methyl,   # [B, L+1], first token is CLS placeholder (0.0 value)
            "positions": pos,        # [B, L+1]
            "special_ids": special,  # [B, L+1] (0=none,1=CLS,2=MASK,3=PAD)
            "targets": targets,      # [B, L+1]
            "loss_mask": loss_mask   # [B, L+1]
        }

def build_dataloader(json_path, cfg_data, cfg_masking, batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2):
    ds = MethylationToCTCFDataset(
        json_path=json_path,
        max_len=cfg_data.get("max_len", 512),
        step=cfg_data.get("step", 1),
        sos_value=cfg_data.get("sos_value", 0.0),
        pad_value=cfg_data.get("pad_value", 0.0),
        normalize_targets=cfg_data.get("normalize_targets", True),
    )
    collate = MaskedValueCollator(
        mlm_prob=cfg_masking.get("mlm_prob", 0.15),
        mask_prob=cfg_masking.get("mask_prob", 0.8),
        random_prob=cfg_masking.get("random_prob", 0.1),
        keep_prob=cfg_masking.get("keep_prob", 0.1),
        random_value_range=tuple(cfg_masking.get("random_value_range", (0.0, 1.0)))
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate, prefetch_factor=prefetch_factor
    )
    return loader
