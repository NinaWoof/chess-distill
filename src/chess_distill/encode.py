import chess
import torch
import numpy as np

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board to a (14, 8, 8) tensor.
    Planes:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Side to move (all 1s if white, all 0s if black) - or relative perspective? 
        We'll keep it simple: perspective is fixed, but planes are always 14.
        Actually, let's use 12 piece planes + 1 side to move + 1 castling/misc.
    """
    # Perspective: we always orient the board so the player to move is "bottom"?
    # No, let's keep it absolute for now and add a plane for side to move.
    
    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
    
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    for i, piece_type in enumerate(pieces):
        # White pieces
        for sq in board.pieces(piece_type, chess.WHITE):
            tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1
        # Black pieces
        for sq in board.pieces(piece_type, chess.BLACK):
            tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1
            
    # Plane 12: Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = -1.0
        
    # Plane 13: Castling rights (simplified)
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13, 0, 0] = 1
    if board.has_kingside_castling_rights(chess.WHITE): tensor[13, 0, 7] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 7, 0] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 7, 7] = 1
    
    return tensor

def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Returns a (8*8*73,) mask of legal moves."""
    from .moves import move_to_index
    mask = torch.zeros(8 * 8 * 73, dtype=torch.bool)
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board.turn)
            mask[idx] = True
        except ValueError:
            continue
    return mask
