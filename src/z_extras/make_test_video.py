import pandas as pd
import numpy as np
import chess
import chess.pgn
import io
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# ==========================================
# CONFIGURATION
# ==========================================
# Paths
RF_PATH = '../models/rf_model.pkl'

LOGREG_PATH = '../models/logreg_model.pkl'
SCALER_PATH = '../models/scaler.pkl'

KNN_PATH = '../models/knn_model_optimized.pkl'
KNN_SCALER_PATH = '../models/scaler_knn.pkl'
KNN_PCA_PATH = '../models/pca_knn.pkl'

DL_PATH = '../models/dl_model_optimized.keras' 
DL_SCALER_PATH = '../models/scaler_dl.pkl'

# OUTPUT NAME (We will try MP4 first)
OUTPUT_FILENAME = 'chess_evolution' 

# Deep Blue vs Kasparov (Game 6)
SAMPLE_PGN = "1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Nd7 5. Ng5 Ngf6 6. Bd3 e6 7. N1f3 h6 8. Nxe6 Qe7 9. O-O fxe6 10. Bg6+ Kd8 11. Bf4 b5 12. a4 Bb7 13. Re1 Nd5 14. Bg3 Kc8 15. axb5 cxb5 16. Qd3 Bc6 17. Bf5 exf5 18. Rxe7 Bxe7 19. c4 1-0"

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

UNICODE_PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
}

def get_board_features(board):
    """Extracts features including MOBILITY"""
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_diff = white_material - black_material

    # Mobility Trick
    current_legal = board.legal_moves.count()
    board.push(chess.Move.null())
    next_legal = board.legal_moves.count()
    board.pop()
    
    if board.turn == chess.WHITE:
        w_mob, b_mob = current_legal, next_legal
    else:
        w_mob, b_mob = next_legal, current_legal
    
    board_state = {}
    for i in range(64):
        piece = board.piece_at(i)
        val = 0
        if piece:
            val = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
        board_state[f'square_{i}'] = val
        
    features = {
        'turn_number': board.fullmove_number,
        'is_white_turn': 1 if board.turn == chess.WHITE else 0,
        'white_rating': 1500, 'black_rating': 1500, 'rating_diff': 0, 
        'white_material': white_material, 'black_material': black_material, 'material_diff': material_diff,
        'white_mobility': w_mob, 'black_mobility': b_mob, 'mobility_diff': w_mob - b_mob,
        'is_check': 1 if board.is_check() else 0
    }
    features.update(board_state)
    return pd.DataFrame([features])

def main():
    print("1. Loading Models safely...")
    assets = {}

    # --- LOAD RANDOM FOREST ---
    if os.path.exists(RF_PATH):
        try:
            assets['rf'] = joblib.load(RF_PATH)
            print("   [OK] Random Forest loaded.")
        except Exception as e: print(f"   [FAIL] RF Error: {e}")
    else:
        print(f"   [SKIP] Random Forest file not found.")

    # --- LOAD LOGISTIC REGRESSION ---
    if os.path.exists(LOGREG_PATH) and os.path.exists(SCALER_PATH):
        try:
            assets['logreg'] = {'model': joblib.load(LOGREG_PATH), 'scaler': joblib.load(SCALER_PATH)}
            print("   [OK] Logistic Regression loaded.")
        except Exception as e: print(f"   [FAIL] LogReg Error: {e}")

    # --- LOAD KNN ---
    if os.path.exists(KNN_PATH) and os.path.exists(KNN_SCALER_PATH) and os.path.exists(KNN_PCA_PATH):
        try:
            assets['knn'] = {'model': joblib.load(KNN_PATH), 'scaler': joblib.load(KNN_SCALER_PATH), 'pca': joblib.load(KNN_PCA_PATH)}
            print("   [OK] KNN loaded.")
        except Exception as e: print(f"   [FAIL] KNN Error: {e}")

    # --- LOAD DEEP LEARNING ---
    actual_dl_path = DL_PATH
    if not os.path.exists(actual_dl_path): actual_dl_path = DL_PATH.replace('.keras', '.h5')
    
    if os.path.exists(actual_dl_path) and os.path.exists(DL_SCALER_PATH):
        try:
            assets['dl'] = {'model': load_model(actual_dl_path), 'scaler': joblib.load(DL_SCALER_PATH)}
            print(f"   [OK] Deep Learning loaded.")
        except Exception as e: print(f"   [FAIL] DL Error: {e}")

    if not assets:
        print("CRITICAL: No models could be loaded. Exiting.")
        return

    print("2. Parsing PGN...")
    pgn = io.StringIO(SAMPLE_PGN)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    
    # Prepare history
    history = {'moves': []}
    for model_name in assets:
        history[model_name] = [0.5]
    history['moves'].append(0)
    
    boards_cache = [board.copy()]

    print("3. Pre-calculating predictions...")
    move_list = list(game.mainline_moves())
    
    for i, move in enumerate(move_list):
        board.push(move)
        boards_cache.append(board.copy())
        X_raw = get_board_features(board)
        
        if 'rf' in assets: history['rf'].append(assets['rf'].predict_proba(X_raw)[0][2])
        if 'logreg' in assets:
            X_s = assets['logreg']['scaler'].transform(X_raw)
            history['logreg'].append(assets['logreg']['model'].predict_proba(X_s)[0][2])
        if 'knn' in assets:
            X_s = assets['knn']['scaler'].transform(X_raw)
            X_p = assets['knn']['pca'].transform(X_s)
            history['knn'].append(assets['knn']['model'].predict_proba(X_p)[0][2])
        if 'dl' in assets:
            X_s = assets['dl']['scaler'].transform(X_raw)
            history['dl'].append(assets['dl']['model'].predict(X_s, verbose=0)[0][2])
            
        history['moves'].append(i + 1)

    # ==============================
    # 4. ANIMATION SETUP
    # ==============================
    print("4. Generating Animation...")
    fig = plt.figure(figsize=(14, 7))
    
    # Grid Layout to make room for Title
    gs = fig.add_gridspec(1, 2)
    ax_board = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    
    def draw_board_on_axis(board, ax):
        ax.clear()
        # Draw checkered board
        for x in range(8):
            for y in range(8):
                color = "#DDB88C" if (x + y) % 2 == 0 else "#A66D4F"
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))
                piece = board.piece_at(chess.square(x, y))
                if piece:
                    symbol = UNICODE_PIECES[piece.symbol()]
                    ax.text(x + 0.5, y + 0.5, symbol, fontsize=35, 
                            ha='center', va='center', color='black')
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([])
        ax.set_yticks([])

    def update(frame):
        # 1. Update Board
        current_board = boards_cache[frame]
        draw_board_on_axis(current_board, ax_board)
        
        # 2. Update Main Title (The Move Counter)
        move_num = history['moves'][frame]
        # Calculate full move number (1, 1, 2, 2, 3, 3...)
        real_move_num = (move_num // 2) + 1
        turn_color = "White" if move_num % 2 == 0 else "Black"
        
        # Overall Title
        fig.suptitle(f"Move {real_move_num}: {turn_color} to Move", fontsize=20, weight='bold')

        # 3. Update Graph
        ax_graph.clear()
        ax_graph.set_ylim(0, 1)
        ax_graph.set_xlim(0, len(move_list))
        ax_graph.set_title("Win Probability (Detect-o-meter)")
        ax_graph.set_ylabel("White Win %")
        ax_graph.set_xlabel("Half-Turns Played")
        
        curr_moves = history['moves'][:frame+1]
        
        if 'rf' in assets: ax_graph.plot(curr_moves, history['rf'][:frame+1], label='Random Forest', color='green', alpha=0.6)
        if 'logreg' in assets: ax_graph.plot(curr_moves, history['logreg'][:frame+1], label='LogReg', color='blue', linestyle='--', alpha=0.6)
        if 'dl' in assets: ax_graph.plot(curr_moves, history['dl'][:frame+1], label='Neural Net', color='red', linewidth=3) # Make NN thickest
        
        ax_graph.axhline(0.5, color='gray', alpha=0.3)
        ax_graph.legend(loc='lower left')
        ax_graph.grid(True, alpha=0.2)

    ani = animation.FuncAnimation(fig, update, frames=len(boards_cache), interval=1000)
    
    # ==============================
    # 5. SAVE (MP4 with Fallback)
    # ==============================
    save_path_mp4 = f"../{OUTPUT_FILENAME}.mp4"
    save_path_gif = f"../{OUTPUT_FILENAME}.gif"
    
    print(f"5. Attempting to save to {save_path_mp4}...")
    try:
        # Try to use FFmpeg for MP4
        writer = animation.FFMpegWriter(fps=1)
        ani.save(save_path_mp4, writer=writer)
        print(f"   SUCCESS! Video saved to {save_path_mp4}")
        print("   (You can pause this video during your presentation!)")
        
    except Exception as e:
        print(f"\n   WARNING: Could not save MP4 (FFmpeg likely missing).")
        print(f"   Error details: {e}")
        print(f"   Falling back to GIF format: {save_path_gif}")
        
        # Fallback to GIF (Pillow is usually built-in)
        writer = animation.PillowWriter(fps=1)
        ani.save(save_path_gif, writer=writer)
        print(f"   SUCCESS! GIF saved to {save_path_gif}")

if __name__ == "__main__":
    main()