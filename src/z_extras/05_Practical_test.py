import pandas as pd
import numpy as np
import chess
import chess.pgn
import io
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURATION
# ==========================================
# Load all the assets we created
RF_PATH = '../models/rf_model.pkl'
LOGREG_PATH = '../models/logreg_model.pkl'
SCALER_PATH = '../models/scaler.pkl'      # For LogReg
KNN_PATH = '../models/knn_model_optimized.pkl'
KNN_SCALER_PATH = '../models/scaler_knn.pkl'
KNN_PCA_PATH = '../models/pca_knn.pkl'

# A Famous Game to Test (Paul Morphy vs Duke of Brunswick, 1858)
SAMPLE_PGN = SAMPLE_PGN = "1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Nd7 5. Ng5 Ngf6 6. Bd3 e6 7. N1f3 h6 8. Nxe6 Qe7 9. O-O fxe6 10. Bg6+ Kd8 11. Bf4 b5 12. a4 Bb7 13. Re1 Nd5 14. Bg3 Kc8 15. axb5 cxb5 16. Qd3 Bc6 17. Bf5 exf5 18. Rxe7 Bxe7 19. c4 1-0"
# Piece values (Same as preprocessing)
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

def get_board_features(board):
    """
    Extracts the EXACT same features as your training script.
    """
    # 1. Material
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_diff = white_material - black_material

    # 2. Meta
    turn = 1 if board.turn == chess.WHITE else 0
    is_check = 1 if board.is_check() else 0
    
    # 3. Board State (64 squares)
    board_state = {}
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
        else:
            val = 0
        board_state[f'square_{i}'] = val
        
    # Combine into a dictionary
    features = {
        'turn_number': board.fullmove_number,
        'is_white_turn': turn,
        # We don't have ratings for a random game, so we assume average/equal (0 diff)
        # This is a limitation, but fine for the visualizer
        'white_rating': 1500,
        'black_rating': 1500,
        'rating_diff': 0, 
        'white_material': white_material,
        'black_material': black_material,
        'material_diff': material_diff,
        'is_check': is_check
    }
    features.update(board_state)
    
    # Return as DataFrame (1 row)
    return pd.DataFrame([features])

def main():
    print("1. Loading Models... (This feels cool)")
    rf_model = joblib.load(RF_PATH)
    logreg_model = joblib.load(LOGREG_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    knn_model = joblib.load(KNN_PATH)
    knn_scaler = joblib.load(KNN_SCALER_PATH)
    knn_pca = joblib.load(KNN_PCA_PATH)
    
    print("2. Replaying the 'Opera Game'...")
    pgn = io.StringIO(SAMPLE_PGN)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    
    # Lists to store probabilities
    rf_probs = []
    logreg_probs = []
    knn_probs = []
    moves = []

    # Iterate through moves
    for move in game.mainline_moves():
        board.push(move)
        moves.append(board.fullmove_number)
        
        # Get raw features
        X_raw = get_board_features(board)
        
        # PREDICT: Random Forest (Raw Data)
        # .predict_proba returns [Prob_Black, Prob_Draw, Prob_White]
        # We want Prob_White (Index 2)
        p_rf = rf_model.predict_proba(X_raw)[0][2]
        rf_probs.append(p_rf)
        
        # PREDICT: Logistic Regression (Scaled)
        X_scaled = scaler.transform(X_raw)
        p_lr = logreg_model.predict_proba(X_scaled)[0][2]
        logreg_probs.append(p_lr)
        
        # PREDICT: KNN (Scaled + PCA)
        X_knn = knn_scaler.transform(X_raw)
        X_knn_pca = knn_pca.transform(X_knn)
        p_knn = knn_model.predict_proba(X_knn_pca)[0][2]
        knn_probs.append(p_knn)

    print("3. Generating the Detect-o-meter Graph...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot lines
    plt.plot(rf_probs, label='Random Forest', color='green', linewidth=2)
    plt.plot(logreg_probs, label='Logistic Regression', color='blue', linestyle='--')
    plt.plot(knn_probs, label='KNN (Optimized)', color='orange', linestyle=':')
    
    # Add the "50% Neutral" line
    plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
    
    plt.title('The "Detect-o-meter": White Win Probability over Time', fontsize=16)
    plt.xlabel('Move Number (Half-Turns)', fontsize=12)
    plt.ylabel('Win Probability (White)', fontsize=12)
    plt.ylim(0, 1) # Probability is always 0 to 1
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the graph
    plt.savefig('../detect_o_meter_result.png')
    print("Done! Check 'detect_o_meter_result.png' in your project folder.")
    plt.show()

if __name__ == "__main__":
    main()