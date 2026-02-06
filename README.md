# Chess Distill

Distilling Stockfish into a high-performance Policy+Value network. Featuring **MCTS Inference**, **Soft Label Distillation**, **Data Augmentation**, and **OneCycleLR** scheduling for Apple Silicon (M4 Pro).

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
# Parallel processing (default: uses all CPU cores - 1)
python scripts/gen_labels.py --pgn_dir data/pgns --out data/dataset.parquet --max_positions 100000

# With explicit worker count
python scripts/gen_labels.py --pgn_dir data/pgns --max_positions 100000 --workers 8

# Single-threaded fallback
python scripts/gen_labels.py --pgn_dir data/pgns --max_positions 100000 --single-thread
```

### 4.b Generate Data (Docker / Remote)
For high-throughput generation on a remote x86 instance:
```bash
# 1. Sync files to instance (fast sync, excludes training data)
rsync -avz --exclude='.venv' --exclude='.git' --exclude='checkpoints' ./ padierna@192.168.86.147:~/ChessBot/chess-distill/

# 2. Build minimal image (no PyTorch, ~250MB)
ssh padierna@192.168.86.147 "cd ~/ChessBot/chess-distill && docker build -t chess-distill-gen ."

# 3. Run with host-drive persistence
# Mounted volume ensures PGNs are read from host and Parquet is saved to host
ssh padierna@192.168.86.147 "docker run -d --name chess-gen -v ~/ChessBot/chess-distill/data:/app/data chess-distill-gen"
```
The labels will be updated at `data/dataset_lichess.parquet` on your remote instance.

### 5. Train
```bash
python scripts/train_policy_value.py --data data/dataset.parquet --epochs 50
```

### 6. Play
```bash
# With MCTS (stronger, default)
python scripts/play_cli.py --ckpt checkpoints/latest.pt --simulations 200

# Without MCTS (faster, weaker)
python scripts/play_cli.py --ckpt checkpoints/latest.pt --no-mcts
```

## Key Features

- **MCTS Inference** (Phase 1): Uses PUCT-based Monte Carlo Tree Search with 200 simulations for stronger move selection.
- **Knowledge Distillation**: Trains on the full MultiPV output (top 10 moves) from Stockfish, using soft targets to capture the engine's "certainty" about a position.
- **Horizontal Symmetry Augmentation**: Automatically flips boards and moves horizontally during training to improve generalization.
- **AdamW + OneCycleLR**: Optimized for Apple Silicon using Metal Performance Shaders (MPS) with an advanced learning rate schedule for faster convergence.
- **Perspective Invariance**: Uses rank flipping to ensure the model learns from a single, standardized rank perspective regardless of color.

## Model Architecture (Phase 1)

| Component | Specification |
|-----------|---------------|
| Backbone | ResNet with **12 residual blocks** |
| Channels | **128** |
| Parameters | ~13.4M |
| Policy Head | 4,672 outputs (AlphaZero-style) |
| Value Head | Single scalar [-1, 1] |

## Training Improvements (Phase 1)

- **Batch size**: 256 (increased for stability)
- **Gradient clipping**: 1.0 (prevents explosions)
- **Data quality**: Stockfish depth 18, 500ms movetime, MultiPV=10

## Data Generation

- **Parallel Processing**: Uses multiprocessing for ~6-10x speedup on multi-core systems
- **Automatic Resume**: Saves progress periodically; restarts from last checkpoint
- **PGN Support**: Extracts positions from real games for higher quality data
