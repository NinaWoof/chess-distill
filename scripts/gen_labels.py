import argparse
import os
import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from chess_distill import config

def wdl_to_value(score: chess.engine.Score) -> float:
    """Converts a score to a value in [-1, 1]."""
    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0
    cp = score.relative.score()
    # Centipawn to scalar: use a sigmoid-like mapping or a simple tanh
    # A common one is: v = torch.tanh(cp / 100)
    return np.tanh(cp / 200.0)

def generate_data(pgn_dir, out_path, max_positions, stockfish_path, depth=12, movetime=None):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    data = []
    positions_seen = set()
    
    if pgn_dir and os.path.exists(pgn_dir):
        pgn_files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith(".pgn")]
        # Support single large PGN if passed as directory or file
        if not pgn_files and pgn_dir.endswith(".pgn"):
            pgn_files = [pgn_dir]
    else:
        pgn_files = []
    
    pbar = tqdm(total=max_positions, desc="Generating positions")
    
    while len(data) < max_positions:
        board = chess.Board()
        
        if pgn_files:
            try:
                # Sample from a random PGN file
                pgn_file = random.choice(pgn_files)
                with open(pgn_file) as f:
                    # To avoid reading massive files, skip to a random offset?
                    # For now, just skip a few games
                    for _ in range(random.randint(0, 100)):
                        if not chess.pgn.skip_game(f):
                            f.seek(0)
                            break
                    game = chess.pgn.read_game(f)
                    
                if game:
                    moves = list(game.mainline_moves())
                    if moves:
                        num_moves = random.randint(0, len(moves) - 1)
                        for i in range(num_moves):
                            board.push(moves[i])
            except Exception as e:
                # Fallback to random if PGN fails
                pass
        
        if board.is_game_over() or len(board.move_stack) == 0:
            # Fallback random playout if PGN sampling didn't happen or we are at start
            for _ in range(random.randint(5, 60)):
                if board.is_game_over(): break
                board.push(random.choice(list(board.legal_moves)))
        
        fen = board.fen()
        if fen in positions_seen: continue
        positions_seen.add(fen)
        
        # Engine analysis
        limit = chess.engine.Limit(depth=depth, time=movetime)
        info = engine.analyse(board, limit, multipv=4)
        
        # Extract best move and value
        best_move = info[0]["pv"][0]
        value = wdl_to_value(info[0]["score"])
        
        # MultiPV targets (soft policy)
        p_targets = []
        for entry in info:
            if "pv" in entry:
                move = entry["pv"][0]
                score = wdl_to_value(entry["score"])
                p_targets.append((move.uci(), score))
        
        data.append({
            "fen": fen,
            "best_move": best_move.uci(),
            "value": value,
            "multi_pv": str(p_targets) # Simple serialization for now
        })
        pbar.update(1)
        
        if len(data) >= max_positions: break
        
    engine.quit()
    df = pd.DataFrame(data)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} positions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--max_positions", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    generate_data(args.pgn_dir, args.out, args.max_positions, config.STOCKFISH_PATH, depth=args.depth)
