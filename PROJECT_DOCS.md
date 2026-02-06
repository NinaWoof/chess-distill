# Chess Distill: Project Documentation

This document serves as a comprehensive reference for the `chess-distill` project, detailing the architecture, implementation choices, and workflow.

## 1. Project Goal
Distill the knowledge of a powerful UCI engine (Stockfish) into a lightweight Policy+Value neural network (PyTorch), optimized for performance on macOS (Apple Silicon M4 Pro).

## 2. System Architecture

### A. Board Representation (`src/chess_distill/encode.py`)
- **Input Channels (14x8x8)**:
  - 6 planes for White pieces (P, N, B, R, Q, K)
  - 6 planes for Black pieces (p, n, b, r, q, k)
  - 1 plane for Side to Move (1.0 for White, -1.0 for Black)
  - 1 plane for Castling Rights (corners marked for specific rights)

### B. Move Encoding (`src/chess_distill/moves.py`)
- **AlphaZero-style (8x8x73)**:
  - Supports all legal chess moves, including promotions.
  - Perspective-aware: Board is mirrored for Black moves so the model learns a standardized representation.
  - Includes `move_to_index` and `index_to_move` for bidirectional mapping.

### C. Model Architecture (`src/chess_distill/model.py`)
- **Backbone**: ResNet with 6 residual blocks and 64 channels.
- **Policy Head**: Outputs logits for 4,672 possible moves.
- **Value Head**: Outputs a single scalar in `[-1, 1]` representing the expected win/loss result.
- **Masking**: Illegal moves are masked out during inference to ensure valid play.

## 3. Data Pipeline (`scripts/gen_labels.py`)
- **Sources**: Random playouts or high-quality games from the **Lichess Elite Database**.
- **Labeling**: Stockfish 18 performs analysis with MultiPV=4 to provide:
  - **Hard/Soft Policy Targets**: The best move (or top 4 moves).
  - **Value Targets**: Centipawn scores converted to a `[-1, 1]` range via `tanh(cp / 200.0)`.

## 4. Training Engine (`src/chess_distill/train.py`)
- **Device Support**: Automatically uses `mps` (Metal Performance Shaders) for high-performance training on Mac hardware.
- **Loss Functions**:
  - `CrossEntropyLoss` for policy.
  - `MSELoss` for value.

## 5. Deployment & Play
- **UCI Protocol (`src/chess_distill/uci.py`)**: Implements standard commands (`position`, `go`, `isready`) allowing the model to be loaded into GUI software like Arena or Cutechess.
- **Interactive CLI (`scripts/play_cli.py`)**: A simple loop to play against the model in the terminal.

## 6. Project History & Decisions
- **Dependency Management**: Standardized on `uv` for lightning-fast environment setup and reproducibility.
- **Deterministic Samples**: The data generator uses `chess.pgn.skip_game` and random offsets to efficiently sample a diverse set of positions from large databases.
- **Environment Safety**: Included `verify_env.py` to ensure Stockfish and MPS are ready before any long-running tasks.

## 7. Key File Map
| File | Purpose |
| :--- | :--- |
| `pyproject.toml` | Dependency definition |
| `src/chess_distill/config.py` | Global constants and device selection |
| `src/chess_distill/model.py` | Neural network definition |
| `scripts/gen_labels.py` | Data collection from Stockfish |
| `scripts/train_policy_value.py` | Training entry point |
| `scripts/play_cli.py` | Human vs Model terminal play |
| `Makefile` | Quick-access commands |

---
*Created by Antigravity (Senior ML Engineer) on February 6th, 2026.*
