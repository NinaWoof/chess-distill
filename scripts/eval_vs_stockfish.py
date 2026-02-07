#!/usr/bin/env python3
"""
Evaluate trained model against Stockfish engine.
Plays multiple games and provides detailed performance metrics.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

import chess
import chess.engine
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chess_distill.infer import load_model
from chess_distill.mcts import MCTS
from chess_distill import config


class GameResult:
    """Store results from a single game."""
    
    def __init__(self, game_num: int, model_color: chess.Color):
        self.game_num = game_num
        self.model_color = model_color
        self.moves: List[str] = []
        self.result: Optional[str] = None
        self.termination: Optional[str] = None
        self.move_times: List[float] = []
        self.pgn: Optional[str] = None
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_num": self.game_num,
            "model_color": "white" if self.model_color == chess.WHITE else "black",
            "result": self.result,
            "termination": self.termination,
            "num_moves": len(self.moves),
            "moves": self.moves,
            "avg_move_time": sum(self.move_times) / len(self.move_times) if self.move_times else 0,
            "pgn": self.pgn
        }


class StockfishEvaluator:
    """Evaluate model by playing against Stockfish."""
    
    def __init__(
        self,
        model_path: str,
        stockfish_path: str = config.STOCKFISH_PATH,
        use_mcts: bool = True,
        mcts_simulations: int = 200,
        stockfish_depth: int = 15,
        stockfish_time: float = 0.5,
        device: str = config.DEVICE
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint
            stockfish_path: Path to Stockfish binary
            use_mcts: Whether to use MCTS for model moves
            mcts_simulations: Number of MCTS simulations per move
            stockfish_depth: Stockfish search depth
            stockfish_time: Stockfish time limit per move (seconds)
            device: Device for model inference
        """
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path, device=device)
        self.device = device
        
        self.use_mcts = use_mcts
        if use_mcts:
            print(f"Initializing MCTS with {mcts_simulations} simulations...")
            self.mcts = MCTS(self.model, device=device, simulations=mcts_simulations, temperature=0)
        else:
            self.mcts = None
            
        print(f"Initializing Stockfish from {stockfish_path}...")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.stockfish_depth = stockfish_depth
        self.stockfish_time = stockfish_time
        
        print("Initialization complete!\n")
        
    def get_model_move(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """Get move from model (with or without MCTS)."""
        start_time = time.time()
        
        if self.mcts:
            move = self.mcts.select_move(board, temperature=0)
            move_time = time.time() - start_time
            return move, move_time
        else:
            from chess_distill.infer import predict
            move, _ = predict(self.model, board, device=self.device)
            move_time = time.time() - start_time
            return move, move_time
    
    def get_stockfish_move(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """Get move from Stockfish."""
        start_time = time.time()
        
        result = self.engine.play(
            board,
            chess.engine.Limit(depth=self.stockfish_depth, time=self.stockfish_time)
        )
        
        move_time = time.time() - start_time
        return result.move, move_time
    
    def play_game(
        self,
        game_num: int,
        model_color: chess.Color,
        verbose: bool = True
    ) -> GameResult:
        """
        Play a single game.
        
        Args:
            game_num: Game number for tracking
            model_color: Color the model plays as
            verbose: Whether to print move-by-move updates
            
        Returns:
            GameResult object with game details
        """
        board = chess.Board()
        result = GameResult(game_num, model_color)
        
        if verbose:
            color_str = "White" if model_color == chess.WHITE else "Black"
            print(f"\n{'='*60}")
            print(f"Game {game_num}: Model plays as {color_str}")
            print(f"{'='*60}\n")
        
        move_num = 1
        while not board.is_game_over():
            # Determine whose turn it is
            is_model_turn = board.turn == model_color
            
            if is_model_turn:
                move, move_time = self.get_model_move(board)
                player = "Model"
            else:
                move, move_time = self.get_stockfish_move(board)
                player = "Stockfish"
            
            # Make the move
            san_move = board.san(move)
            board.push(move)
            result.moves.append(san_move)
            result.move_times.append(move_time)
            
            if verbose:
                move_display = f"{move_num}. " if board.turn == chess.BLACK else f"{move_num}... "
                print(f"{move_display}{san_move:8} ({player}, {move_time:.2f}s)")
                
            if board.turn == chess.WHITE:
                move_num += 1
        
        # Record result
        outcome = board.outcome()
        if outcome:
            result.result = outcome.result()
            result.termination = outcome.termination.name
            
            if verbose:
                print(f"\nGame Over: {result.result}")
                print(f"Termination: {result.termination}")
                
                # Determine winner
                if result.result == "1-0":
                    winner = "Model" if model_color == chess.WHITE else "Stockfish"
                elif result.result == "0-1":
                    winner = "Model" if model_color == chess.BLACK else "Stockfish"
                else:
                    winner = "Draw"
                print(f"Winner: {winner}\n")
        
        # Generate PGN
        result.pgn = self._generate_pgn(board, result)
        
        return result
    
    def _generate_pgn(self, board: chess.Board, game_result: GameResult) -> str:
        """Generate PGN string for the game."""
        import chess.pgn
        
        game = chess.pgn.Game()
        game.headers["Event"] = "Model vs Stockfish Evaluation"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "ChessDistill" if game_result.model_color == chess.WHITE else "Stockfish"
        game.headers["Black"] = "Stockfish" if game_result.model_color == chess.WHITE else "ChessDistill"
        game.headers["Result"] = game_result.result or "*"
        
        node = game
        temp_board = chess.Board()
        for san_move in game_result.moves:
            move = temp_board.parse_san(san_move)
            node = node.add_variation(move)
            temp_board.push(move)
        
        return str(game)
    
    def run_evaluation(
        self,
        num_games: int,
        alternate_colors: bool = True,
        verbose: bool = True,
        save_results: bool = True,
        output_dir: str = "eval_results"
    ) -> Dict:
        """
        Run full evaluation across multiple games.
        
        Args:
            num_games: Number of games to play
            alternate_colors: Whether to alternate model color each game
            verbose: Whether to print detailed game info
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation statistics
        """
        results: List[GameResult] = []
        
        print(f"\n{'#'*60}")
        print(f"Starting Evaluation: {num_games} games")
        print(f"Model: {'MCTS' if self.use_mcts else 'Direct Policy'}")
        print(f"Stockfish: Depth {self.stockfish_depth}, Time {self.stockfish_time}s")
        print(f"{'#'*60}\n")
        
        for i in range(num_games):
            # Determine model color
            if alternate_colors:
                model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            else:
                model_color = chess.WHITE
            
            # Play game
            game_result = self.play_game(i + 1, model_color, verbose=verbose)
            results.append(game_result)
        
        # Calculate statistics
        stats = self._calculate_stats(results)
        
        # Print summary
        self._print_summary(stats)
        
        # Save results
        if save_results:
            self._save_results(results, stats, output_dir)
        
        return stats
    
    def _calculate_stats(self, results: List[GameResult]) -> Dict:
        """Calculate evaluation statistics."""
        total_games = len(results)
        wins = 0
        losses = 0
        draws = 0
        
        for result in results:
            if result.result == "1-0":
                if result.model_color == chess.WHITE:
                    wins += 1
                else:
                    losses += 1
            elif result.result == "0-1":
                if result.model_color == chess.BLACK:
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1
        
        win_rate = wins / total_games * 100 if total_games > 0 else 0
        draw_rate = draws / total_games * 100 if total_games > 0 else 0
        loss_rate = losses / total_games * 100 if total_games > 0 else 0
        
        # Average game length
        avg_moves = sum(len(r.moves) for r in results) / total_games if total_games > 0 else 0
        
        # Average move time
        all_times = [t for r in results for t in r.move_times]
        avg_move_time = sum(all_times) / len(all_times) if all_times else 0
        
        return {
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "avg_moves_per_game": avg_moves,
            "avg_move_time": avg_move_time
        }
    
    def _print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Games:     {stats['total_games']}")
        print(f"Wins:            {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"Draws:           {stats['draws']} ({stats['draw_rate']:.1f}%)")
        print(f"Losses:          {stats['losses']} ({stats['loss_rate']:.1f}%)")
        print(f"Avg Game Length: {stats['avg_moves_per_game']:.1f} moves")
        print(f"Avg Move Time:   {stats['avg_move_time']:.3f}s")
        print(f"{'='*60}\n")
    
    def _save_results(self, results: List[GameResult], stats: Dict, output_dir: str):
        """Save results to files."""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON summary
        json_path = output_path / f"eval_{timestamp}.json"
        data = {
            "timestamp": timestamp,
            "config": {
                "use_mcts": self.use_mcts,
                "mcts_simulations": self.mcts.simulations if self.mcts else 0,
                "stockfish_depth": self.stockfish_depth,
                "stockfish_time": self.stockfish_time
            },
            "statistics": stats,
            "games": [r.to_dict() for r in results]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {json_path}")
        
        # Save PGN file with all games
        pgn_path = output_path / f"games_{timestamp}.pgn"
        with open(pgn_path, 'w') as f:
            for result in results:
                f.write(result.pgn)
                f.write("\n\n")
        
        print(f"PGN games saved to {pgn_path}")
    
    def close(self):
        """Clean up resources."""
        if self.engine:
            self.engine.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chess model against Stockfish engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play 10 games with MCTS (200 simulations) vs Stockfish depth 15
  python eval_vs_stockfish.py --ckpt checkpoints/latest.pt --games 10
  
  # Fast evaluation: no MCTS, weaker Stockfish
  python eval_vs_stockfish.py --ckpt checkpoints/latest.pt --games 20 --no-mcts --sf-depth 10
  
  # Strong evaluation: more MCTS simulations, stronger Stockfish
  python eval_vs_stockfish.py --ckpt checkpoints/latest.pt --games 5 --simulations 400 --sf-depth 20 --sf-time 1.0
        """
    )
    
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play (default: 10)")
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS (use direct policy)")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per move (default: 200)")
    parser.add_argument("--sf-depth", type=int, default=15, help="Stockfish search depth (default: 15)")
    parser.add_argument("--sf-time", type=float, default=0.5, help="Stockfish time per move in seconds (default: 0.5)")
    parser.add_argument("--sf-path", type=str, default=config.STOCKFISH_PATH, help="Path to Stockfish binary")
    parser.add_argument("--no-alternate", action="store_true", help="Don't alternate colors (model always plays white)")
    parser.add_argument("--quiet", action="store_true", help="Suppress move-by-move output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory for results (default: eval_results)")
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not Path(args.ckpt).exists():
        print(f"Error: Checkpoint not found at {args.ckpt}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = StockfishEvaluator(
        model_path=args.ckpt,
        stockfish_path=args.sf_path,
        use_mcts=not args.no_mcts,
        mcts_simulations=args.simulations,
        stockfish_depth=args.sf_depth,
        stockfish_time=args.sf_time
    )
    
    try:
        # Run evaluation
        evaluator.run_evaluation(
            num_games=args.games,
            alternate_colors=not args.no_alternate,
            verbose=not args.quiet,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
