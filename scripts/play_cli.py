import chess
import argparse
from chess_distill.infer import load_model, predict
from chess_distill.mcts import MCTS
from chess_distill import config

def play(ckpt_path, human_color=chess.WHITE, use_mcts=True, simulations=config.MCTS_SIMULATIONS):
    model = load_model(ckpt_path)
    board = chess.Board()
    
    # Initialize MCTS if enabled
    mcts = MCTS(model, simulations=simulations) if use_mcts else None
    mode_str = f"MCTS ({simulations} sims)" if use_mcts else "Greedy"
    print(f"Starting game. Model mode: {mode_str}")
    
    while not board.is_game_over():
        print("\n", board)
        if board.turn == human_color:
            move_str = input(f"\nYour move ({board.turn}): ")
            try:
                move = board.parse_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move!")
            except:
                print("Invalid UCI string!")
        else:
            if mcts:
                move = mcts.select_move(board, temperature=0)
                val = 0.0
            else:
                move, val = predict(model, board)
            print(f"\nModel move: {move} (Eval: {val:.4f})")
            board.push(move)
            
    print("\nGame Over!")
    print("Result:", board.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--black", action="store_true", help="Play as black")
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS (use greedy move selection)")
    parser.add_argument("--simulations", type=int, default=config.MCTS_SIMULATIONS, help="MCTS simulations")
    args = parser.parse_args()
    
    color = chess.BLACK if args.black else chess.WHITE
    play(args.ckpt, color, use_mcts=not args.no_mcts, simulations=args.simulations)
