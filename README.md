# Chess Distill

Distilling Stockfish into a Policy+Value network using PyTorch on macOS (M4 Pro).

## Quickstart

### 1. Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Stockfish (`brew install stockfish`)

### 2. Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Verify Environment
```bash
python scripts/verify_env.py
```

### 4. Generate Data
```bash
python scripts/gen_labels.py --pgn_dir data/pgns --out data/dataset.parquet --max_positions 1000
```

### 5. Train
```bash
python scripts/train_policy_value.py --data data/dataset.parquet --epochs 2
```

### 6. Play
```bash
python scripts/play_cli.py --ckpt checkpoints/latest.pt
```
