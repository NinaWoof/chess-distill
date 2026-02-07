import chess
import torch
import numpy as np

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board to a (14, 8, 8) tensor.
    Planes:
    0-5: Friendly pieces (P, N, B, R, Q, K)
    6-11: Opponent pieces (p, n, b, r, q, k)
    12: Side to move (all 1s if white, all -1s if black - standard logic, 
        but we mirror the board so this is less critical now)
    13: Castling rights (marked in corners)
    
    NEW: Mirroring ensures "Friendly" is always at the bottom rank perspective.
    """
    tensor = torch.zeros((16, 8, 8), dtype=torch.float32)
    
    # We use 'us' and 'them' for side-to-move invariance
    us = board.turn
    them = not us
    
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    for i, piece_type in enumerate(pieces):
        # Our pieces
        for sq in board.pieces(piece_type, us):
            # If black, flip rank
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            if us == chess.BLACK:
                rank = 7 - rank
                # file = 7 - file  # Should we also flip file? Standard AlphaZero flips ranks. 
                # Actually Standard AlphaZero flips for Black.
            tensor[i, rank, file] = 1
            
        # Their pieces
        for sq in board.pieces(piece_type, them):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            if us == chess.BLACK:
                rank = 7 - rank
            tensor[i + 6, rank, file] = 1
            
    # Plane 12: Side to move - constant 1.0 since board is already perspective-invariant
    # Using -1.0 for Black is redundant and can cause issues with ReLU activations
    tensor[12, :, :] = 1.0
        
    # Plane 13: Castling rights (mirrored)
    # White view: (0,0) (0,7). Black view: (7,0) (7,7) mirrored to (0,0) (0,7)
    if us == chess.WHITE:
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, 0, 0] = 1
        if board.has_kingside_castling_rights(chess.WHITE): tensor[13, 0, 7] = 1
        if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 7, 0] = 1
        if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 7, 7] = 1
    else:
        # For Black to move, Black is "us"
        if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 0, 0] = 1
        if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 0, 7] = 1
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, 7, 0] = 1
        if board.has_kingside_castling_rights(chess.WHITE): tensor[13, 7, 7] = 1
    
    # Plane 14: En Passant (target square)
    if board.ep_square:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        if us == chess.BLACK:
            rank = 7 - rank
        tensor[14, rank, file] = 1.0
        
    # Plane 15: Half-move clock (normalized)
    tensor[15, :, :] = float(board.halfmove_clock) / 100.0
    
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
