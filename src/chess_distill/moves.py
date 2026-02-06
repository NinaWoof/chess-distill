import chess
import numpy as np

# 8x8x73 encoding
# 73 planes:
# 56 queen moves (7 squares * 8 directions)
# 8 knight moves
# 9 pawn promotions (3 directions * 3 pieces: Knight, Bishop, Rook. Queen is included in queen moves)
# We'll use a simpler map for now: 
# 0-55: Queen-like moves (distance 1-7 in 8 directions: N, NE, E, SE, S, SW, W, NW)
# 56-63: Knight moves
# 64-66: Under-promotions to Knight (N, NE, NW)
# 67-69: Under-promotions to Bishop
# 70-72: Under-promotions to Rook

# Actually, the 73 planes are usually:
# 8 directions * 7 squares = 56
# 8 knight directions = 8
# 9 promotions = 3 directions * 3 pieces (N, B, R)
# 56 + 8 + 9 = 73

def move_to_index(move: chess.Move, color: chess.Color) -> int:
    """Encodes a chess.Move into an index in [0, 8*8*73)."""
    from_sq = move.from_square
    to_sq = move.to_square
    
    # If black, flip the board vertically for a standardized perspective
    if color == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    
    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)
    
    dr = to_rank - from_rank
    df = to_file - from_file
    
    plane = -1
    
    # Knight moves
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    if (dr, df) in knight_moves:
        plane = 56 + knight_moves.index((dr, df))
    
    # Under-promotions
    elif move.promotion and move.promotion != chess.QUEEN:
        promo_type = move.promotion
        # Only happens on the last rank (rank 7 after flip)
        # Directions: straight (df=0), right (df=1), left (df=-1)
        direction_idx = df + 1 # 0: left, 1: straight, 2: right
        if promo_type == chess.KNIGHT:
            plane = 64 + direction_idx
        elif promo_type == chess.BISHOP:
            plane = 67 + direction_idx
        elif promo_type == chess.ROOK:
            plane = 70 + direction_idx
    
    # Queen-like moves (includes queen promotions)
    else:
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        abs_dr, abs_df = abs(dr), abs(df)
        if dr == 0 or df == 0 or abs_dr == abs_df:
            # It's a straight or diagonal move
            dist = max(abs_dr, abs_df)
            dir_dr = dr // dist if dr != 0 else 0
            dir_df = df // dist if df != 0 else 0
            dir_idx = directions.index((dir_dr, dir_df))
            plane = dir_idx * 7 + (dist - 1)
            
    if plane == -1:
        raise ValueError(f"Could not encode move {move}")
        
    return (plane * 64) + from_sq

def index_to_move(idx: int, color: chess.Color, board: chess.Board) -> chess.Move:
    """Decodes an index into a chess.Move."""
    from_sq = idx % 64
    plane = idx // 64
    
    orig_from_sq = from_sq
    if color == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        
    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    
    # Decode plane
    if plane < 56: # Queen-like
        directions = [ (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1) ]
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr, df = directions[dir_idx]
        if color == chess.BLACK: dr = -dr
        to_rank, to_file = from_rank + dr * dist, from_file + df * dist
    elif plane < 64: # Knight
        knight_moves = [ (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1) ]
        dr, df = knight_moves[plane - 56]
        if color == chess.BLACK: dr = -dr
        to_rank, to_file = from_rank + dr, from_file + df
    else: # Promotions
        direction_idx = (plane - 64) % 3
        df = direction_idx - 1
        dr = 1 if color == chess.WHITE else -1
        to_rank, to_file = from_rank + dr, from_file + df
        
    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None # Illegal move (off board)
        
    to_sq = chess.square(to_file, to_rank)
    
    # Handle promotion
    promo = None
    if plane >= 64:
        if plane < 67: promo = chess.KNIGHT
        elif plane < 70: promo = chess.BISHOP
        elif plane < 73: promo = chess.ROOK
    else:
        # Check for auto-queen promotion if it reaches last rank
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (color == chess.WHITE and to_rank == 7) or (color == chess.BLACK and to_rank == 0):
                promo = chess.QUEEN
                
    return chess.Move(from_sq, to_sq, promotion=promo)
