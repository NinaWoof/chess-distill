import os
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import shutil

# Engine Config
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH") or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"

# Model Config
BOARD_SIZE = 8
CHANNELS = 128  # Phase 1: Scaled from 64
RES_BLOCKS = 12  # Phase 1: Scaled from 6
POLICY_SIZE = 8 * 8 * 73  # AlphaZero style

# Training Config
if HAS_TORCH:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    DEVICE = "cpu"
BATCH_SIZE = 256  # Phase 1: Increased for stability
LEARNING_RATE = 1e-3
GRADIENT_CLIP = 1.0  # Phase 1: Prevent gradient explosion

# MCTS Config (Phase 1)
MCTS_SIMULATIONS = 200
MCTS_CPUCT = 1.5  # Exploration constant
MCTS_TEMPERATURE = 1.0  # Move selection temperature

# Data Config
DEFAULT_DATA_PATH = "data/dataset.parquet"
CHECKPOINT_DIR = "checkpoints"
