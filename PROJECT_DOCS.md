# Chess Distill: Project Documentation

This document serves as a comprehensive reference for the `chess-distill` project, detailing the architecture, implementation choices, and workflow.

## 1. Project Goal
Distill the knowledge of a powerful UCI engine (Stockfish) into a lightweight Policy+Value neural network (PyTorch), optimized for performance on macOS (Apple Silicon M4 Pro).

## 2. System Architecture

### A. Board Representation (`src/chess_distill/encode.py`)
- **Input Channels (16x8x8)**:
  - 6 planes for Friendly pieces (P, N, B, R, Q, K)
  - 6 planes for Opponent pieces (p, n, b, r, q, k)
  - 1 plane for Side to Move (constant 1.0 - perspective-invariant)
  - 1 plane for Castling Rights
  - 1 plane for En Passant
  - 1 plane for Halfmove Clock (normalized)
- **Perspective Invariance**: Board is always flipped so the active player is at the bottom (rank mirroring for Black).

### B. Move Encoding (`src/chess_distill/moves.py`)
- **AlphaZero-style (8x8x73)**:
  - Supports all legal chess moves, including promotions.
  - Perspective-aware: Board is mirrored for Black moves.
  - **Horizontal Symmetry**: The dataset loader supports random horizontal flipping for data augmentation.

### C. Model Architecture (`src/chess_distill/model.py`)
- **Backbone**: ResNet with **12 residual blocks** and **128 channels** (~13.4M parameters).
- **Policy Head**: Outputs logits for 4,672 possible moves.
- **Value Head**: Outputs a single scalar in `[-1, 1]`.

### D. MCTS Inference (`src/chess_distill/mcts.py`)
- **PUCT Algorithm**: Uses policy network as prior, value network for evaluation.
- **Default**: 200 simulations per move (configurable: 800-1600 for stronger play).
- **Selection**: UCB1 with exploration constant (c_puct = 1.5).
- **Temperature**: Configurable for training (exploratory) vs play (greedy).
- **Value Backup**: Perspective-invariant (no value negation during backup).

## 3. Data Pipeline (`scripts/gen_labels.py`)
- **Sources**: PGN databases (Lichess, etc.) or random playouts.
- **Labeling**: Stockfish performs analysis with **MultiPV=10** at **depth 18**.
- **Soft Policy Targets**: Full probability distribution of top 10 moves (Soft Labels).
- **Value Targets**: Centipawn scores converted to `[-1, 1]` via `tanh(cp / 200.0)`.
- **Parallel Processing**: Uses `multiprocessing.Pool` with multiple Stockfish workers.
  - Default: `cpu_count - 1` workers for ~6-10x speedup.
  - Each worker spawns its own Stockfish instance.
  - Configurable via `--workers N` flag.
  - Fallback to single-thread with `--single-thread` flag.
- **Automatic Resume**: Periodic saves allow resuming from interruptions.
- **Dockerized Generation**: Optimized for remote x86 instances.
  - **Minimal Dependencies**: Uses a custom build without PyTorch to save space and build time.
  - **Volume Persistence**: Data is mounted from the host drive (`data/` -> `/app/data`).
  - **Resource Scaling**: Uses `cpu_count - 1` workers by default inside the container.

## 4. Training Engine (`src/chess_distill/train.py`)
- **Metal Acceleration**: Uses `mps` for Apple Silicon (M4 Pro).
- **Optimizer**: `AdamW` with `OneCycleLR` scheduler.
- **Batch Size**: 256 (Phase 1 scaling).
- **Gradient Clipping**: Max norm 1.0 for stability.
- **Loss Functions**:
  - `CrossEntropyLoss` with soft targets for policy distillation.
  - `MSELoss` for value.

## 5. Deployment & Play
- **UCI Protocol (`src/chess_distill/uci.py`)**: Implements standard commands with MCTS support. Configurable via `setoption`.
- **Interactive CLI (`scripts/play_cli.py`)**: Play against the model with optional MCTS.

## 6. Evaluation & Diagnostics
- **Stockfish Evaluation (`scripts/eval_vs_stockfish.py`)**: Automated testing against Stockfish with configurable depth, time limits, and MCTS simulations.
  - Outputs JSON results and PGN games for analysis.
  - Supports quiet mode for batch testing.
- **Move Diagnostics (`scripts/diagnose_moves.py`)**: Tests move encoding/decoding symmetry, board encoding consistency, and model predictions.
- **MCTS Testing (`scripts/test_mcts.py`)**: Verifies MCTS move selection and visit counts.

## 7. Project History & Decisions
- **Phase 1 (Feb 2026)**: Scaled model to 12×128, added MCTS, improved data quality.
- **Bug Fix (Feb 7, 2026)**: Fixed MCTS value backup bug causing inverted move preferences. Model now plays proper openings (d4, Nf3, e4) instead of suicidal moves (h4, Rh3).
- **Dependency Management**: Standardized on `uv` for lightning-fast environment setup.
- **Deterministic Samples**: The data generator uses random offsets to sample diverse positions.
- **Environment Safety**: Included `verify_env.py` to ensure Stockfish and MPS are ready.

## 8. Key File Map
| File | Purpose |
| :--- | :--- |
| `pyproject.toml` | Dependency definition |
| `src/chess_distill/config.py` | Global constants, model config, MCTS settings |
| `src/chess_distill/model.py` | Neural network definition |
| `src/chess_distill/mcts.py` | Monte Carlo Tree Search |
| `scripts/gen_labels.py` | Parallel data collection from Stockfish |
| `scripts/train_policy_value.py` | Training entry point |
| `scripts/play_cli.py` | Human vs Model terminal play |
| `scripts/eval_vs_stockfish.py` | Automated evaluation against Stockfish |
| `scripts/diagnose_moves.py` | Move encoding and model prediction diagnostics |
| `scripts/test_mcts.py` | MCTS move selection testing |
| `src/chess_distill/dataset.py` | PyTorch dataset with augmentation |
| `Makefile` | Quick-access commands |

---
*Updated for Phase 1 — February 7th, 2026.*
