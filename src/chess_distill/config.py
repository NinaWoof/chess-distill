import os
import torch
import shutil

# Engine Config
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH") or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"

# Model Config
BOARD_SIZE = 8
CHANNELS = 64
RES_BLOCKS = 6
POLICY_SIZE = 8 * 8 * 73  # AlphaZero style

# Training Config
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Data Config
DEFAULT_DATA_PATH = "data/dataset.parquet"
CHECKPOINT_DIR = "checkpoints"
