#!/usr/bin/env python3
"""Test MCTS move selection to see if it's causing the h4 bug."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from chess_distill.infer import load_model
from chess_distill.mcts import MCTS

def test_mcts_selection():
    print("="*60)
    print("Testing MCTS Move Selection")
    print("="*60)
    
    model = load_model("checkpoints/best.pt")
    board = chess.Board()
    
    # Test with MCTS
    mcts = MCTS(model, simulations=200, temperature=0)
    
    print("\nRunning MCTS with 200 simulations...")
    visit_counts = mcts.search(board)
    
    # Sort by visit count
    sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 moves by visit count:")
    print(f"{'Rank':<6} {'Move':<8} {'Visits':<10} {'UCI':<8}")
    print("-" * 40)
    
    for rank, (move, visits) in enumerate(sorted_moves[:10], 1):
        san = board.san(move)
        print(f"{rank:<6} {san:<8} {visits:<10} {move.uci():<8}")
    
    # Check h4
    h4_move = chess.Move.from_uci("h2h4")
    if h4_move in visit_counts:
        h4_visits = visit_counts[h4_move]
        h4_rank = sum(1 for v in visit_counts.values() if v > h4_visits) + 1
        print(f"\nMove 'h4' rank: {h4_rank} (visits: {h4_visits})")
    else:
        print("\nMove 'h4' not in visit counts")
    
    # Select best move
    best_move = mcts.select_move(board, temperature=0)
    print(f"\nMCTS selected move: {board.san(best_move)} ({best_move.uci()})")

if __name__ == "__main__":
    test_mcts_selection()
