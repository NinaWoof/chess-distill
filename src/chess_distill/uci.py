import sys
import chess
from .infer import load_model, predict

def uci_loop(ckpt_path):
    model = load_model(ckpt_path)
    board = chess.Board()
    
    while True:
        line = sys.stdin.readline()
        if not line: break
        cmd = line.strip().split()
        if not cmd: continue
        
        if cmd[0] == "uci":
            print("id name ChessDistill")
            print("id author SeniorML")
            print("uciok")
        elif cmd[0] == "isready":
            print("readyok")
        elif cmd[0] == "ucinewgame":
            board = chess.Board()
        elif cmd[0] == "position":
            if cmd[1] == "startpos":
                board = chess.Board()
                moves_idx = 2
            elif cmd[1] == "fen":
                fen = " ".join(cmd[2:8])
                board = chess.Board(fen)
                moves_idx = 8
            
            if "moves" in cmd:
                moves_idx = cmd.index("moves") + 1
                for m in cmd[moves_idx:]:
                    board.push_uci(m)
        elif cmd[0] == "go":
            move, val = predict(model, board)
            print(f"bestmove {move.uci()}")
        elif cmd[0] == "quit":
            break
        sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    uci_loop(args.ckpt)
