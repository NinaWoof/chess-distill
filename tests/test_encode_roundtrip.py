import chess
from chess_distill.moves import move_to_index, index_to_move

def test_move_roundtrip():
    board = chess.Board()
    # E2E4
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move, chess.WHITE)
    decoded = index_to_move(idx, chess.WHITE, board)
    assert decoded == move
    
    # Black move (with flip)
    board.push(move)
    move_black = chess.Move.from_uci("e7e5")
    idx_black = move_to_index(move_black, chess.BLACK)
    decoded_black = index_to_move(idx_black, chess.BLACK, board)
    assert decoded_black == move_black

    # Knight move
    move_kn = chess.Move.from_uci("g1f3")
    idx_kn = move_to_index(move_kn, chess.WHITE)
    decoded_kn = index_to_move(idx_kn, chess.WHITE, board)
    assert decoded_kn == move_kn

if __name__ == "__main__":
    test_move_roundtrip()
    print("Test passed!")
