import chess
import torch
from chess_distill.encode import board_to_tensor
from chess_distill.moves import move_to_index

def test_perspective_consistency():
    # 1. Start with a standard position
    board_w = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1")
    tensor_w = board_to_tensor(board_w)
    
    # 2. Mirror the board manually (simplified: just flip ranks and swap pieces)
    # Actually, let's use a position that is naturally symmetric or just flip it.
    # A mirrored version of 1. e4 (White) is 1... e5 (Black) from Black's view.
    board_b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR b KQkq e6 0 1")
    tensor_b = board_to_tensor(board_b)
    
    # Check piece planes 0-5 (Friendly)
    # The mirrored e4 and e5 should look identical to the model
    # (White's P on e4 vs Black's P on e5)
    diff = torch.abs(tensor_w[0:12] - tensor_b[0:12]).sum()
    print(f"Piece planes diff: {diff.item()}")
    
    if diff < 1e-5:
        print("SUCCESS: Perspectives are consistent!")
    else:
        print("FAILURE: Perspectives inconsistent.")
        
    # Check move encoding consistency
    move_w = chess.Move.from_uci("e2e4")
    move_b = chess.Move.from_uci("e7e5")
    
    idx_w = move_to_index(move_w, chess.WHITE)
    idx_b = move_to_index(move_b, chess.BLACK)
    
    print(f"White move index: {idx_w}")
    print(f"Black move index: {idx_b}")
    
    if idx_w == idx_b:
        print("SUCCESS: Move indices are consistent!")
    else:
        print("FAILURE: Move indices inconsistent.")

if __name__ == "__main__":
    test_perspective_consistency()
