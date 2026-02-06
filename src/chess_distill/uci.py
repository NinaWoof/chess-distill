import sys
import chess
from .infer import load_model, predict
from .mcts import MCTS
from . import config

def uci_loop(ckpt_path, use_mcts=True, simulations=config.MCTS_SIMULATIONS):
    model = load_model(ckpt_path)
    board = chess.Board()
    
    # Initialize MCTS if enabled
    mcts = MCTS(model, simulations=simulations) if use_mcts else None
    
    while True:
        line = sys.stdin.readline()
        if not line: break
        cmd = line.strip().split()
        if not cmd: continue
        
        if cmd[0] == "uci":
            print("id name ChessDistill-MCTS")
            print("id author SeniorML")
            print(f"option name MCTS type check default {'true' if use_mcts else 'false'}")
            print(f"option name Simulations type spin default {simulations} min 1 max 10000")
            print("uciok")
        elif cmd[0] == "isready":
            print("readyok")
        elif cmd[0] == "setoption":
            # Parse setoption name X value Y
            if "name" in cmd and "value" in cmd:
                name_idx = cmd.index("name") + 1
                value_idx = cmd.index("value") + 1
                opt_name = cmd[name_idx]
                opt_value = cmd[value_idx]
                if opt_name.lower() == "mcts":
                    use_mcts = opt_value.lower() == "true"
                    mcts = MCTS(model, simulations=simulations) if use_mcts else None
                elif opt_name.lower() == "simulations":
                    simulations = int(opt_value)
                    if mcts:
                        mcts.simulations = simulations
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
            if mcts:
                # Use MCTS for move selection (temperature 0 = best move)
                move = mcts.select_move(board, temperature=0)
                val = 0.0  # MCTS doesn't return single value
            else:
                move, val = predict(model, board)
            print(f"bestmove {move.uci()}")
        elif cmd[0] == "quit":
            break
        sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS (greedy mode)")
    parser.add_argument("--simulations", type=int, default=config.MCTS_SIMULATIONS)
    args = parser.parse_args()
    uci_loop(args.ckpt, use_mcts=not args.no_mcts, simulations=args.simulations)
