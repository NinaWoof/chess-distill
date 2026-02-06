import chess.engine
from chess_distill import config

def test_stockfish_query():
    engine = chess.engine.SimpleEngine.popen_uci(config.STOCKFISH_PATH)
    board = chess.Board()
    info = engine.analyse(board, chess.engine.Limit(depth=10))
    
    assert "score" in info[0]
    assert "pv" in info[0]
    
    engine.quit()
    print("Stockfish query test passed!")

if __name__ == "__main__":
    test_stockfish_query()
