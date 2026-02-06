import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
from .encode import board_to_tensor
from .moves import move_to_index
import ast

class ChessDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = chess.Board(row["fen"])
        
        x = board_to_tensor(board)
        
        # Target policy: currently just the best move as index
        best_move_obj = chess.Move.from_uci(row["best_move"])
        target_move_idx = move_to_index(best_move_obj, board.turn)
        
        target_value = torch.tensor([row["value"]], dtype=torch.float32)
        
        return x, target_move_idx, target_value
