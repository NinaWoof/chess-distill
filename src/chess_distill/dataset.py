import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
from .encode import board_to_tensor
from .moves import move_to_index
from . import config
import ast
import re

def parse_multi_pv(multi_pv_str):
    """Parse multi_pv string, handling both clean and np.float64 formats."""
    try:
        # Try standard parsing first
        return ast.literal_eval(multi_pv_str)
    except (ValueError, SyntaxError):
        # Handle np.float64(...) format from old data
        # Pattern: ('move', np.float64(value))
        pattern = r"\('([^']+)',\s*np\.float64\(([-\d.e+]+)\)\)"
        matches = re.findall(pattern, multi_pv_str)
        if matches:
            return [(move, float(score)) for move, score in matches]
        return []

class ChessDataset(Dataset):
    def __init__(self, parquet_path, augment=True, temperature=1.0):
        self.df = pd.read_parquet(parquet_path)
        self.augment = augment
        self.temperature = temperature
        
    def __len__(self):
        return len(self.df)
    
    def _mirror_move(self, move: chess.Move) -> chess.Move:
        """Mirrors a move horizontally."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Horizontal mirror: file -> 7 - file
        new_from = chess.square(7 - chess.square_file(from_sq), chess.square_rank(from_sq))
        new_to = chess.square(7 - chess.square_file(to_sq), chess.square_rank(to_sq))
        
        return chess.Move(new_from, new_to, promotion=move.promotion)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = chess.Board(row["fen"])
        
        # Soft targets preparation - use robust parser
        multi_pv = parse_multi_pv(row["multi_pv"])
        
        # Augmentation: Horizontal Flip
        flip = self.augment and (torch.rand(1).item() > 0.5)
        if flip:
            board = board.transform(chess.flip_horizontal)
            # Flip multiPV entries
            new_pv = []
            for move_uci, score in multi_pv:
                m = chess.Move.from_uci(move_uci)
                m_mirrored = self._mirror_move(m)
                new_pv.append((m_mirrored.uci(), score))
            multi_pv = new_pv
            
        x = board_to_tensor(board)
        
        # Target policy: Soft Labels (MultiPV)
        target_policy = torch.zeros(config.POLICY_SIZE, dtype=torch.float32)
        
        scores = []
        indices = []
        for move_uci, cp_score in multi_pv:
            try:
                m = chess.Move.from_uci(move_uci)
                move_idx = move_to_index(m, board.turn)
                indices.append(move_idx)
                # Centipawn to logits: we use a simpler scaling
                scores.append(cp_score / (100.0 * self.temperature))
            except:
                continue
        
        if scores:
            scores_t = torch.tensor(scores, dtype=torch.float32)
            probs = torch.softmax(scores_t, dim=0)
            for i, move_idx in enumerate(indices):
                target_policy[move_idx] = probs[i]
        else:
            # Fallback to hard label if MultiPV is empty/broken
            best_move_obj = chess.Move.from_uci(row["best_move"])
            if flip: best_move_obj = self._mirror_move(best_move_obj)
            target_move_idx = move_to_index(best_move_obj, board.turn)
            target_policy[target_move_idx] = 1.0
        
        target_value = torch.tensor([row["value"]], dtype=torch.float32)
        
        return x, target_policy, target_value
