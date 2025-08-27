
import json
import torch
from torch.utils.data import Dataset
import orjson

class MethylationToCTCFDataset(Dataset):
    def __init__(self, json_path, max_len=512, step=1, sos_value=0.0, pad_value=0.0, normalize_targets=True):
        with open(json_path, "rb") as f:
            # self.data = json.load(f)
            self.data = orjson.loads(f.read())
        self.max_len = max_len
        self.step = step
        self.sos_value = sos_value
        self.pad_value = pad_value
        self.normalize_targets = normalize_targets

    def _build_positions(self, item, seq_len):
        # If positions array is provided, use it. Otherwise, generate from start position with step size.
        if "positions" in item and isinstance(item["positions"], list):
            pos = item["positions"][:seq_len]
        else:
            start = int(item["position"]) if "position" in item else 0
            pos = list(range(start, start + seq_len * self.step, self.step))
        return pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


#  ["chr", "start", "end","methyValue", "ctcfValue", "sequence"]

        it = self.data[idx]

        seq = it["methyValue"][: self.max_len]
        tgt = it["ctcfValue"][: self.max_len]

        positions = self._build_positions(it, len(seq))

        L = len(seq)
        if L < self.max_len:
            pad_n = self.max_len - L
            seq = seq + [0.0] * pad_n
            tgt = tgt + [0.0] * pad_n
            positions = positions + [positions[-1] + self.step * i for i in range(1, pad_n + 1)]

        y = torch.tensor(tgt, dtype=torch.float32)

        # Decoder input: <SOS> + shifted target
        dec_in = torch.tensor([self.sos_value] + tgt[:-1], dtype=torch.float32)

        # Build mask for PAD positions
        valid_len = min(len(it["ctcfValue"]), self.max_len)
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
