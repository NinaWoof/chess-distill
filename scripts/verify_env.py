import sys
import torch
import chess.engine
from chess_distill import config

def verify():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    stockfish_path = config.STOCKFISH_PATH
    print(f"Looking for Stockfish at: {stockfish_path}")
    
    if not stockfish_path or not os.path.exists(stockfish_path):
        print("❌ Stockfish not found! Set STOCKFISH_PATH or 'brew install stockfish'")
        sys.exit(1)
        
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print(f"✅ Stockfish (UCI) working: {engine.id.get('name')}")
        engine.quit()
    except Exception as e:
        print(f"❌ Failed to start Stockfish: {e}")
        sys.exit(1)
        
    print("✅ Environment verified.")

if __name__ == "__main__":
    import os
    verify()
