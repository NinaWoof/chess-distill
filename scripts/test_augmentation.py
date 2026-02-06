import chess
import torch
from chess_distill.encode import board_to_tensor
from chess_distill.moves import move_to_index
from chess_distill.dataset import ChessDataset

def test_mirroring():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    # Dataset just for the helper
    dataset = ChessDataset.__new__(ChessDataset)
    dataset.augment = False
    move = chess.Move.from_uci("f3e5") # White knight takes e5
    mirrored_move = dataset._mirror_move(move)
    print(f"Original: {move.uci()} -> Mirrored: {mirrored_move.uci()}")
    
    # Mirror f3e5 around d/e file: 
    # f (5) -> 7-5 = 2 (c)
    # e (4) -> 7-4 = 3 (d)
    # Result: c3d5? wait.
    # f3 is (file 5, rank 2). Mirror(5,2) -> (2,2) which is c3.
    # e5 is (file 4, rank 4). Mirror(4,4) -> (3,4) which is d5.
    # Correct: c3d5.
    
    assert mirrored_move.uci() == "c3d5"
    print("Move mirroring: OK")
    
    # 2. Test board mirroring
    board_mirrored = board.transform(chess.flip_horizontal)
    print(f"Original FEN: {board.fen()}")
    print(f"Mirrored FEN: {board_mirrored.fen()}")
    
    # 3. Test reversibility
    board_double_mirrored = board_mirrored.transform(chess.flip_horizontal)
    assert board.fen() == board_double_mirrored.fen()
    print("Board mirroring reversibility: OK")

if __name__ == "__main__":
    test_mirroring()
