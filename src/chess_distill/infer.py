import torch
import chess
from .model import ChessNet
from .encode import board_to_tensor, get_legal_move_mask
from .moves import index_to_move
from . import config

def load_model(ckpt_path, device=config.DEVICE):
    model = ChessNet()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, board, device=config.DEVICE):
    x = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        p_logits, v = model(x)
    
    # Mask illegal moves
    mask = get_legal_move_mask(board).to(device)
    p_logits[0][~mask] = -float('inf')
    
    # Best move
    best_move_idx = torch.argmax(p_logits[0]).item()
    best_move = index_to_move(best_move_idx, board.turn, board)
    
    return best_move, v.item()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    model = load_model(args.ckpt)
    board = chess.Board(args.fen)
    move, val = predict(model, board)
    print(f"Move: {move} | Value: {val:.4f}")
