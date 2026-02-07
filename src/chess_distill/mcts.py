"""
Monte Carlo Tree Search (MCTS) with PUCT algorithm.
Phase 1 implementation for improved move selection.
"""

import math
import chess
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from . import config
from .model import ChessNet
from .encode import board_to_tensor, get_legal_move_mask
from .moves import index_to_move, move_to_index


class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, 
                 prior: float = 0.0, move: Optional[chess.Move] = None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # P(s, a) from policy network
        
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """Mean value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, cpuct: float = config.MCTS_CPUCT) -> float:
        """
        UCB1 score with policy prior (PUCT formula from AlphaZero).
        Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        exploration = cpuct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.value + exploration
    
    def select_child(self) -> 'MCTSNode':
        """Select the child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand(self, policy_probs: torch.Tensor):
        """Expand node using policy network output."""
        self.is_expanded = True
        
        for move in self.board.legal_moves:
            try:
                move_idx = move_to_index(move, self.board.turn)
                prior = policy_probs[move_idx].item()
            except ValueError:
                prior = 1e-6  # Small prior for unmapped moves
                
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, parent=self, prior=prior, move=move)
    
    def backup(self, value: float):
        """Backpropagate value through the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            # Since board encoding is perspective-invariant (always "us" at bottom),
            # the value is already from the current player's POV at each node.
            # No need to flip the value.
            node.value_sum += value
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""
    
    def __init__(self, model: ChessNet, device: str = config.DEVICE,
                 simulations: int = config.MCTS_SIMULATIONS,
                 cpuct: float = config.MCTS_CPUCT,
                 temperature: float = config.MCTS_TEMPERATURE):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.cpuct = cpuct
        self.temperature = temperature
        
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """Get policy and value from neural network."""
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        
        # Apply legal move mask
        legal_mask = get_legal_move_mask(board).to(self.device)
        policy_logits = policy_logits.squeeze(0)
        policy_logits[~legal_mask] = -float('inf')
        
        # Softmax for probabilities
        policy_probs = torch.softmax(policy_logits, dim=0)
        
        return policy_probs.cpu(), value.item()
    
    def search(self, root_board: chess.Board) -> Dict[chess.Move, float]:
        """
        Run MCTS from the given position.
        Returns: Dictionary mapping moves to visit count proportions.
        """
        root = MCTSNode(root_board)
        
        # Expand root
        policy_probs, _ = self._evaluate(root_board)
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(self.simulations):
            node = root
            
            # Selection: traverse to a leaf
            while node.is_expanded and node.children:
                node = node.select_child()
                if node is None:
                    break
            
            # Check for terminal state
            if node.board.is_game_over():
                result = node.board.result()
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = -1.0 if node.board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0
            else:
                # Expansion and evaluation
                policy_probs, value = self._evaluate(node.board)
                if node.board.legal_moves:
                    node.expand(policy_probs)
            
            # Backup
            node.backup(value)
        
        # Calculate visit count distribution
        visit_counts = {}
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count
            
        return visit_counts
    
    def select_move(self, board: chess.Board, temperature: Optional[float] = None) -> chess.Move:
        """
        Select a move using MCTS.
        
        Args:
            board: Current position
            temperature: If 0, select best move. If > 0, sample proportionally.
        
        Returns:
            Selected move
        """
        if temperature is None:
            temperature = self.temperature
            
        visit_counts = self.search(board)
        
        if not visit_counts:
            # Fallback to random legal move
            return list(board.legal_moves)[0]
        
        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
        
        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(counts)
            return moves[best_idx]
        else:
            # Temperature-based sampling
            counts = counts ** (1 / temperature)
            probs = counts / counts.sum()
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
    
    def get_policy_target(self, board: chess.Board) -> torch.Tensor:
        """
        Get improved policy target from MCTS visit counts.
        Used for self-play training (Phase 2+).
        """
        visit_counts = self.search(board)
        
        policy = torch.zeros(config.POLICY_SIZE, dtype=torch.float32)
        total_visits = sum(visit_counts.values())
        
        if total_visits > 0:
            for move, count in visit_counts.items():
                try:
                    idx = move_to_index(move, board.turn)
                    policy[idx] = count / total_visits
                except ValueError:
                    continue
                    
        return policy
