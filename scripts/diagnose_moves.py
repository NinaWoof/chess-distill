#!/usr/bin/env python3
"""
Diagnostic script to test move encoding/decoding and model predictions.
This will help identify coordinate/perspective mismatches.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import torch
from chess_distill.encode import board_to_tensor, get_legal_move_mask
from chess_distill.moves import move_to_index, index_to_move
from chess_distill.infer import load_model, predict


def test_move_encoding():
    """Test that move encoding/decoding is symmetric."""
    print("="*60)
    print("TEST 1: Move Encoding/Decoding Symmetry")
    print("="*60)
    
    board = chess.Board()  # Starting position
    
    # Test common opening moves
    test_moves = [
        chess.Move.from_uci("e2e4"),  # e4
        chess.Move.from_uci("d2d4"),  # d4
        chess.Move.from_uci("g1f3"),  # Nf3
        chess.Move.from_uci("c2c4"),  # c4
    ]
    
    print("\nTesting White moves from starting position:")
    for move in test_moves:
        try:
            # Encode
            idx = move_to_index(move, chess.WHITE)
            # Decode
            decoded = index_to_move(idx, chess.WHITE, board)
            
            match = "✓" if decoded == move else "✗"
            print(f"{match} {move.uci():6} -> idx {idx:5} -> {decoded.uci() if decoded else 'None':6}")
            
            if decoded != move:
                print(f"   ERROR: Expected {move.uci()}, got {decoded.uci() if decoded else 'None'}")
        except Exception as e:
            print(f"✗ {move.uci():6} -> ERROR: {e}")
    
    # Test Black moves
    board_black = chess.Board()
    board_black.push(chess.Move.from_uci("e2e4"))  # 1. e4
    
    test_moves_black = [
        chess.Move.from_uci("e7e5"),  # e5
        chess.Move.from_uci("c7c5"),  # c5
        chess.Move.from_uci("g8f6"),  # Nf6
    ]
    
    print("\nTesting Black moves after 1. e4:")
    for move in test_moves_black:
        try:
            idx = move_to_index(move, chess.BLACK)
            decoded = index_to_move(idx, chess.BLACK, board_black)
            
            match = "✓" if decoded == move else "✗"
            print(f"{match} {move.uci():6} -> idx {idx:5} -> {decoded.uci() if decoded else 'None':6}")
            
            if decoded != move:
                print(f"   ERROR: Expected {move.uci()}, got {decoded.uci() if decoded else 'None'}")
        except Exception as e:
            print(f"✗ {move.uci():6} -> ERROR: {e}")


def test_model_predictions(ckpt_path):
    """Test what the model actually predicts for the starting position."""
    print("\n" + "="*60)
    print("TEST 2: Model Predictions on Starting Position")
    print("="*60)
    
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    model = load_model(ckpt_path)
    board = chess.Board()
    
    # Get model's device
    device = next(model.parameters()).device
    
    # Get model's raw policy output
    x = board_to_tensor(board).unsqueeze(0).to(device)
    
    with torch.no_grad():
        p_logits, v = model(x)
    
    # Apply legal move mask
    legal_mask = get_legal_move_mask(board).to(p_logits.device)
    p_logits = p_logits.squeeze(0)
    p_logits[~legal_mask] = -float('inf')
    
    # Get top 10 moves
    probs = torch.softmax(p_logits, dim=0)
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)
    
    print(f"\nTop {top_k} moves predicted by model:")
    print(f"{'Rank':<6} {'Move':<8} {'Probability':<12} {'UCI':<8}")
    print("-" * 40)
    
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        try:
            move = index_to_move(idx.item(), board.turn, board)
            if move and move in board.legal_moves:
                san = board.san(move)
                print(f"{rank:<6} {san:<8} {prob.item():<12.4f} {move.uci():<8}")
            else:
                print(f"{rank:<6} {'ILLEGAL':<8} {prob.item():<12.4f} {'N/A':<8}")
        except Exception as e:
            print(f"{rank:<6} {'ERROR':<8} {prob.item():<12.4f} {str(e)[:20]}")
    
    print(f"\nValue prediction: {v.item():.4f}")
    
    # Check if h4 is in top moves
    h4_move = chess.Move.from_uci("h2h4")
    try:
        h4_idx = move_to_index(h4_move, chess.WHITE)
        h4_prob = probs[h4_idx].item()
        h4_rank = (probs > h4_prob).sum().item() + 1
        print(f"\nMove 'h4' rank: {h4_rank} (probability: {h4_prob:.4f})")
    except:
        print("\nCould not find h4 in move encoding")
    
    # Expected good moves
    print("\nExpected good opening moves (e4, d4, Nf3, c4):")
    good_moves = ["e2e4", "d2d4", "g1f3", "c2c4"]
    for uci in good_moves:
        try:
            move = chess.Move.from_uci(uci)
            idx = move_to_index(move, chess.WHITE)
            prob = probs[idx].item()
            rank = (probs > prob).sum().item() + 1
            print(f"  {board.san(move):<6} rank: {rank:<4} prob: {prob:.4f}")
        except Exception as e:
            print(f"  {uci:<6} ERROR: {e}")


def test_board_encoding():
    """Test board encoding for both colors."""
    print("\n" + "="*60)
    print("TEST 3: Board Encoding Consistency")
    print("="*60)
    
    board = chess.Board()
    
    # Encode for White
    tensor_white = board_to_tensor(board)
    print(f"\nWhite to move - tensor shape: {tensor_white.shape}")
    print(f"Plane 12 (side to move): unique values = {tensor_white[12].unique().tolist()}")
    
    # Check piece positions
    print("\nWhite pieces (plane 0-5):")
    for i, piece_name in enumerate(['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']):
        count = tensor_white[i].sum().item()
        print(f"  {piece_name}: {int(count)} pieces")
    
    # Make a move and test Black
    board.push(chess.Move.from_uci("e2e4"))
    tensor_black = board_to_tensor(board)
    print(f"\nBlack to move - tensor shape: {tensor_black.shape}")
    print(f"Plane 12 (side to move): unique values = {tensor_black[12].unique().tolist()}")
    
    print("\nBlack pieces (plane 0-5, should be 'us' now):")
    for i, piece_name in enumerate(['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']):
        count = tensor_black[i].sum().item()
        print(f"  {piece_name}: {int(count)} pieces")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose move encoding and model predictions")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="Model checkpoint")
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("# CHESS MODEL DIAGNOSTIC TOOL")
    print("#"*60)
    
    # Run tests
    test_move_encoding()
    test_board_encoding()
    test_model_predictions(args.ckpt)
    
    print("\n" + "#"*60)
    print("# DIAGNOSTIC COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
