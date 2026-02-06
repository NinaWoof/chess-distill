import chess
import argparse
from chess_distill.infer import load_model, predict

def play(ckpt_path, human_color=chess.WHITE):
    model = load_model(ckpt_path)
    board = chess.Board()
    
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
            move, val = predict(model, board)
            print(f"\nModel move: {move} (Eval: {val:.4f})")
            board.push(move)
            
    print("\nGame Over!")
    print("Result:", board.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--black", action="store_true", help="Play as black")
    args = parser.parse_args()
    
    color = chess.BLACK if args.black else chess.WHITE
    play(args.ckpt, color)
