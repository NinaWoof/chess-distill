#!/usr/bin/env python3
"""
Multi-process data generation for chess distillation.
Uses parallel Stockfish instances for faster position labeling.
"""

import argparse
import os
import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count
import signal
from chess_distill import config

# Global engine instance for workers
_worker_engine = None

def wdl_to_value(score: chess.engine.PovScore) -> float:
    """Converts a PovScore to a value in [-1, 1] from the perspective of the side to move."""
    rel = score.relative
    if rel.is_mate():
        return 1.0 if rel.mate() > 0 else -1.0
    cp = rel.score()
    return np.tanh(cp / 200.0)

def sample_position_from_pgn(pgn_files):
    """Sample a random position from PGN files."""
    if not pgn_files:
        return None
    
    board = chess.Board()
    try:
        pgn_file = random.choice(pgn_files)
        with open(pgn_file) as f:
            file_size = os.path.getsize(pgn_file)
            f.seek(random.randint(0, max(0, file_size - 1024*1024)))
            while True:
                line = f.readline()
                if not line or line.startswith("[Event "): break
            
            game = chess.pgn.read_game(f)
            
        if game:
            moves = list(game.mainline_moves())
            if moves:
                num_moves = random.randint(0, len(moves) - 1)
                for i in range(num_moves):
                    board.push(moves[i])
                return board
    except Exception:
        pass
    return None

def random_position():
    """Generate a random position via random play."""
    board = chess.Board()
    for _ in range(random.randint(5, 60)):
        if board.is_game_over(): break
        board.push(random.choice(list(board.legal_moves)))
    return board

def init_worker(stockfish_path):
    """Initialize a persistent engine for each worker process."""
    global _worker_engine
    # Ignore signals in workers to let the main process handle them
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"Worker failed to start Stockfish at {stockfish_path}: {e}")
        _worker_engine = None

def worker_analyze(fen, depth, movetime, multipv):
    """Analyze a single position using the cached worker engine."""
    global _worker_engine
    if _worker_engine is None:
        return None
    
    try:
        board = chess.Board(fen)
        limit = chess.engine.Limit(depth=depth, time=movetime/1000.0)
        info = _worker_engine.analyse(board, limit, multipv=multipv)
        
        if not info or "pv" not in info[0] or not info[0]["pv"]:
            return None
            
        best_move = info[0]["pv"][0]
        value = wdl_to_value(info[0]["score"])
        
        p_targets = []
        for entry in info:
            if "pv" in entry and entry["pv"]:
                move = entry["pv"][0]
                score = entry["score"].relative
                if score.is_mate():
                    val = 10000 if score.mate() > 0 else -10000
                else:
                    val = score.score()
                p_targets.append((move.uci(), val))
        
        if not p_targets:
            return None
            
        return {
            "fen": fen,
            "best_move": best_move.uci(),
            "value": value,
            "multi_pv": str(p_targets)
        }
    except Exception as e:
        # Silent error in worker, but we could log it
        return None

def generate_data_multiprocess(pgn_dir, out_path, max_positions, stockfish_path, 
                               depth=18, movetime=500, multipv=10, num_workers=None):
    """Generate data using multiple Stockfish processes."""
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Using {num_workers} parallel workers with persistent Stockfish instances.")
    
    data = []
    positions_seen = set()
    
    # Resume from existing data
    if os.path.exists(out_path):
        try:
            old_df = pd.read_parquet(out_path)
            data = old_df.to_dict("records")
            positions_seen = set(old_df["fen"].tolist())
            print(f"Resuming from {len(data)} existing positions.")
        except:
            pass
    
    # Find PGN files
    pgn_files = []
    if pgn_dir and os.path.exists(pgn_dir):
        if os.path.isdir(pgn_dir):
            pgn_files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith(".pgn")]
        elif pgn_dir.endswith(".pgn"):
            pgn_files = [pgn_dir]
    
    pbar = tqdm(total=max_positions, desc="Generating positions", initial=len(data))
    batch_size = num_workers * 10  # Larger batches for better throughput
    
    # Initialize pool once with persistent engines
    with Pool(processes=num_workers, initializer=init_worker, initargs=(stockfish_path,)) as pool:
        while len(data) < max_positions:
            # Generate a batch of unique positions
            fens_to_analyze = []
            attempts = 0
            while len(fens_to_analyze) < batch_size and attempts < batch_size * 10:
                attempts += 1
                
                if pgn_files:
                    board = sample_position_from_pgn(pgn_files)
                    if board is None or board.is_game_over() or len(board.move_stack) == 0:
                        board = random_position()
                else:
                    board = random_position()
                
                fen = board.fen()
                if fen not in positions_seen:
                    positions_seen.add(fen)
                    fens_to_analyze.append((fen, depth, movetime, multipv))
            
            if not fens_to_analyze:
                print("Struggling to find unique FENs, retrying...")
                continue
            
            # Analyze in parallel
            results = pool.starmap(worker_analyze, fens_to_analyze)
            
            # Collect results
            batch_added = 0
            for result in results:
                if result is not None:
                    data.append(result)
                    pbar.update(1)
                    batch_added += 1
                    
                    if len(data) >= max_positions:
                        break
            
            print(f"Batch complete. Added {batch_added}/{len(fens_to_analyze)} positions. Total: {len(data)}", flush=True)

            # Periodic save
            if len(data) % 100 < batch_size:
                pd.DataFrame(data).to_parquet(out_path)
                print(f"Checkpoint saved to {out_path}", flush=True)
    
    pbar.close()
    df = pd.DataFrame(data)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} positions to {out_path}")

def generate_data(pgn_dir, out_path, max_positions, stockfish_path, depth=18, movetime=500, multipv=10):
    """Original single-threaded implementation."""
    print("Running in single-threaded mode...")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    data = []
    positions_seen = set()
    
    if os.path.exists(out_path):
        try:
            old_df = pd.read_parquet(out_path)
            data = old_df.to_dict("records")
            positions_seen = set(old_df["fen"].tolist())
            print(f"Resuming from {len(data)} existing positions.")
        except:
            pass

    pgn_files = []
    if pgn_dir and os.path.exists(pgn_dir):
        pgn_files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith(".pgn")]
        if not pgn_files and pgn_dir.endswith(".pgn"):
            pgn_files = [pgn_dir]
    
    pbar = tqdm(total=max_positions, desc="Generating positions", initial=len(data))
    
    while len(data) < max_positions:
        board = chess.Board()
        
        if pgn_files:
            try:
                pgn_file = random.choice(pgn_files)
                with open(pgn_file) as f:
                    file_size = os.path.getsize(pgn_file)
                    f.seek(random.randint(0, max(0, file_size - 1024*1024)))
                    while True:
                        line = f.readline()
                        if not line or line.startswith("[Event "): break
                    
                    game = chess.pgn.read_game(f)
                    
                if game:
                    moves = list(game.mainline_moves())
                    if moves:
                        num_moves = random.randint(0, len(moves) - 1)
                        for i in range(num_moves):
                            board.push(moves[i])
            except Exception:
                pass
        
        if board.is_game_over() or len(board.move_stack) == 0:
            board = random_position()
        
        fen = board.fen()
        if fen in positions_seen: continue
        positions_seen.add(fen)
        
        limit = chess.engine.Limit(depth=depth, time=movetime/1000.0)
        try:
            info = engine.analyse(board, limit, multipv=multipv)
            
            if not info or "pv" not in info[0] or not info[0]["pv"]:
                continue
                
            best_move = info[0]["pv"][0]
            value = wdl_to_value(info[0]["score"])
            
            p_targets = []
            for entry in info:
                if "pv" in entry and entry["pv"]:
                    move = entry["pv"][0]
                    score = entry["score"].relative
                    if score.is_mate():
                        val = 10000 if score.mate() > 0 else -10000
                    else:
                        val = score.score()
                    p_targets.append((move.uci(), val))
            
            if not p_targets:
                continue

            data.append({
                "fen": fen,
                "best_move": best_move.uci(),
                "value": value,
                "multi_pv": str(p_targets)
            })
            pbar.update(1)
            
            if len(data) % 500 == 0:
                pd.DataFrame(data).to_parquet(out_path)
        except Exception:
            continue
        
    engine.quit()
    df = pd.DataFrame(data)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} positions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--max_positions", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=18, help="Stockfish analysis depth")
    parser.add_argument("--movetime", type=int, default=500, help="Stockfish movetime in ms")
    parser.add_argument("--multipv", type=int, default=10, help="Number of PV lines for soft labels")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument("--single-thread", action="store_true", help="Use single-threaded mode")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    if args.single_thread:
        generate_data(args.pgn_dir, args.out, args.max_positions, config.STOCKFISH_PATH,
                      depth=args.depth, movetime=args.movetime, multipv=args.multipv)
    else:
        generate_data_multiprocess(args.pgn_dir, args.out, args.max_positions, config.STOCKFISH_PATH,
                                   depth=args.depth, movetime=args.movetime, multipv=args.multipv,
                                   num_workers=args.workers)
