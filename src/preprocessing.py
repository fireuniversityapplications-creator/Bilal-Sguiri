import pandas as pd
import chess
import numpy as np
from tqdm import tqdm  # This creates a progress bar
import ast

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = '../data/raw/games.csv'       # Name of your lichess dataset
OUTPUT_FILE = '../data/processed/master_data.csv'

# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def parse_game_data(row):
    """
    Takes a single row (one game) and breaks it down into 
    multiple rows (one per move/position).
    """
    
    # 1. SETUP THE BOARD
    # ------------------
    moves_string = row['moves'] # Adjust column name if your CSV is different
    try:
        moves = moves_string.split()
    except AttributeError:
        # Handle cases where moves_string might be NaN
        return []

    board = chess.Board()
    
    # Determine the final result (Target Label)
    # Mapping: 1 = White Win, 0 = Black Win, 0.5 = Draw
    if row['winner'] == 'white':
        game_result = 1
    elif row['winner'] == 'black':
        game_result = 0
    else:
        game_result = 0.5 

    game_data = []

    # 2. REPLAY THE GAME (The Loop)
    # -----------------------------
    # We iterate through every move in the game history
    for move in moves:
        try:
            # Push the move to the virtual board
            board.push_san(move)
        except ValueError:
            # If an illegal move is found in the CSV, stop parsing this game
            break
        
        # 3. EXTRACT FEATURES (The Snapshot)
        # ----------------------------------
        
        # A. Material Count
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        material_diff = white_material - black_material

        # B. Board State (64 Squares)
        # We create a dictionary for the board state
        board_state = {}
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                # piece_type is 1-6. Multiplied by 1 for white, -1 for black
                val = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
            else:
                val = 0
            board_state[f'square_{i}'] = val

        # C. Meta-Data (Turn, Check, etc)
        is_check = 1 if board.is_check() else 0
        turn = 1 if board.turn == chess.WHITE else 0 # 1 for White's turn

        # D. MOBILITY (The Curse Breaker)
        # -------------------------------
        # Calculate how many legal moves the current player has.
        # Then, temporarily pass the turn to see how many moves the opponent has.
        
        # 1. Moves for the player whose turn it is right now
        current_legal_moves = board.legal_moves.count()
        
        # 2. Switch turn (using a Null Move) to count opponent's moves
        board.push(chess.Move.null())
        next_legal_moves = board.legal_moves.count()
        board.pop() # Restore the board state
        
        if board.turn == chess.WHITE:
            white_mobility = current_legal_moves
            black_mobility = next_legal_moves
        else:
            white_mobility = next_legal_moves
            black_mobility = current_legal_moves
            
        mobility_diff = white_mobility - black_mobility
        
        # Create the data row
        snapshot = {
            'game_id': row['id'],
            'turn_number': board.fullmove_number,
            'is_white_turn': turn,
            'white_rating': row['white_rating'],
            'black_rating': row['black_rating'],
            'rating_diff': row['white_rating'] - row['black_rating'],
            'white_material': white_material,
            'black_material': black_material,
            'material_diff': material_diff,
            'white_mobility': white_mobility, # NEW
            'black_mobility': black_mobility, # NEW
            'mobility_diff': mobility_diff,   # NEW
            'is_check': is_check,
            'result': game_result
        }
        
        # Merge the board state (64 columns) into the snapshot
        snapshot.update(board_state)
        
        game_data.append(snapshot)

    return game_data

def main():
    print("Loading raw data...")
    # Read the CSV. 
    df = pd.read_csv(INPUT_FILE)
    
    # PHASE 1: INITIAL CLEANING
    print(f"Original games: {len(df)}")
    
    # Filter short games
    df = df[df['turns'] >= 10]
    print(f"Games after filtering short games: {len(df)}")

    # PHASE 2 & 3: PARSING AND FEATURE EXTRACTION
    print("Parsing games with Mobility features... This may take 10-15 minutes.")
    
    all_positions = []
    
    # tqdm gives us a nice progress bar in the terminal
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            game_positions = parse_game_data(row)
            all_positions.extend(game_positions)
        except Exception as e:
            # print(f"Error parsing game {row['id']}: {e}")
            continue

    print(f"Processed {len(all_positions)} total board positions.")
    
    # Convert list of dicts to DataFrame
    final_df = pd.DataFrame(all_positions)
    
    # Save to CSV
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Master dataset created.")

if __name__ == "__main__":
    main()