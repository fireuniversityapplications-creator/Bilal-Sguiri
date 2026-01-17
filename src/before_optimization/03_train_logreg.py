import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ==========================================
# 1. HYPERPARAMETER CONFIGURATION
# ==========================================
LOGREG_CONFIG = {
    'solver': 'saga',        # 'saga' is fast for large datasets
    'max_iter': 200,         # Increase if it doesn't converge
    'n_jobs': -1,            # Use all CPUs
    'C': 0.1,                # Regularization (Lower = Stronger penalty against noise)
    'random_state': 42
}

# ==========================================
# 2. FILE PATHS
# ==========================================
INPUT_FILE = '../data/processed/master_data.csv'
MODEL_PATH = '../models/logreg_model.pkl'
SCALER_PATH = '../models/scaler.pkl'
PLOT_DIR = '../plots'

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/logreg_confusion_matrix.png')
    plt.close()
    print(f"   Saved confusion matrix to {PLOT_DIR}/logreg_confusion_matrix.png")

def plot_coefficients(model, feature_names):
    """
    Plots the top 15 features that influence winning.
    Positive bars = Helps White Win.
    Negative bars = Helps Black Win.
    """
    # For multiclass, coef_ is shape (3, n_features). 
    # Index 2 corresponds to "White Win" class coefficients vs the rest.
    coeffs = model.coef_[2] 
    
    indices = np.argsort(np.abs(coeffs))[::-1] # Sort by magnitude (absolute value)
    top_n = 15
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Features Influencing White Victory (LogReg)')
    plt.barh(range(top_n), coeffs[top_indices], align='center', color='blue')
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Coefficient Value (Positive = Good for White)')
    plt.axvline(x=0, color='black', linestyle='--') # Center line
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/logreg_coefficients.png')
    plt.close()
    print(f"   Saved coefficients plot to {PLOT_DIR}/logreg_coefficients.png")

def main():
    print("1. Loading Master Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # -------------------------------------------------------
    # DATA SPLITTING
    # -------------------------------------------------------
    print("2. Splitting Data by Game ID...")
    unique_games = df['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_games)
    
    split_idx = int(len(unique_games) * 0.8)
    train_game_ids = unique_games[:split_idx]
    test_game_ids = unique_games[split_idx:]
    
    train_df = df[df['game_id'].isin(train_game_ids)]
    test_df = df[df['game_id'].isin(test_game_ids)]
    
    print(f"   Training on {len(train_game_ids)} games")
    print(f"   Testing on  {len(test_game_ids)} games")

    drop_cols = ['game_id', 'result']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['result']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['result']

    label_mapping = {0.0: 0, 0.5: 1, 1.0: 2}
    y_train = y_train.map(label_mapping).astype(int)
    y_test = y_test.map(label_mapping).astype(int)

    # -------------------------------------------------------
    # SCALING
    # -------------------------------------------------------
    print("3. Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   Scaling Complete.")

    # -------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------
    print(f"4. Training Logistic Regression with config: {LOGREG_CONFIG}...")
    
    logreg = LogisticRegression(**LOGREG_CONFIG)
    logreg.fit(X_train_scaled, y_train)
    print("   Training Complete.")

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------
    print("5. Evaluating Model...")
    y_pred = logreg.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Black Win', 'Draw', 'White Win']))

    # -------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------
    print("6. Generating Presentation Plots...")
    plot_confusion_matrix(y_test, y_pred, classes=['Black Win', 'Draw', 'White Win'])
    plot_coefficients(logreg, X_train.columns)

    # -------------------------------------------------------
    # SAVE
    # -------------------------------------------------------
    print(f"7. Saving model and scaler...")
    joblib.dump(logreg, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("   Done.")

if __name__ == "__main__":
    main()