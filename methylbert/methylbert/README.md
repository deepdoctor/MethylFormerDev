# MethylBERT — BERT-style Pretraining for Methylation Coverage

This project pretrains a Transformer encoder on 1D methylation coverage series using a masked-value modeling objective.
It follows the BERT masking recipe (15% mask; 80/10/10 replace policy) and predicts the original methylation value only at masked positions.
A learnable `[CLS]` token is prepended to every sequence (usable for sequence-level downstream tasks).

## Quick Start

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train with default config
python train.py --config configs/default.yaml

# 3) Override settings from CLI
python train.py --config configs/default.yaml training.batch_size=32 model.d_model=256
```

## Data Format

The dataloader expects a JSON list of items. Each item should include at least:
```json
{
  "chr": "chr1",
  "start": 10000,
  "end": 10000 + 512,
  "methyValue": [0.0, 0.1, 0.2, ...],  // length can be <= max_len (will be padded)
  "ctcfValue": [0.0, 0.0, ...],        // ignored during pretraining
  "positions": [10000, 10001, ...]     // optional; if missing, generated from "position" and step
}
```
Your dataset class is included (with minor adjustments for reuse), but pretraining uses `methyValue` as both input and target.

## Files

- `configs/default.yaml` — All hyperparameters (model, masking, optimizer, training).
- `data.py` — Dataset, collator with BERT-style masking, and DataLoader builders.
- `model.py` — Transformer encoder with learnable `[CLS]`, `[MASK]`, `[PAD]` embeddings; regression head.
- `train.py` — Training loop with checkpointing and gradient accumulation.
- `utils.py` — Utilities (seed, config system, logging).
- `requirements.txt` — Dependencies.

## Notes

- Loss is computed **only** on masked tokens (Huber or MSE; configurable).
- Positions (genomic coordinates) are projected by a small MLP and added to value embeddings.
- `[CLS]` participates in attention but is excluded from the regression loss.
- FP16 (AMP) supported.
