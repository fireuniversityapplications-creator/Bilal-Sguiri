import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ==========================================
# 1. HYPERPARAMETER CONFIGURATION
# ==========================================
# Tweak these to optimize your model!
RF_CONFIG = {
    'n_estimators': 100,       # Number of trees (Higher = Better but slower)
    'max_depth': 40,           # How deep each tree can grow (Higher = More complex patterns)
    'min_samples_split': 2,    # Minimum samples required to split a node (Prevents overfitting)
    'max_features': 0.5,      # Look at 80% of features at every split (very aggressive)
    'n_jobs': -1,              # Use all CPU cores
    'random_state': 42         # Fix random seed for reproducible results
}

# ==========================================
# 2. FILE PATHS
# ==========================================
INPUT_FILE = '../data/processed/master_data.csv'
MODEL_PATH = '../models/rf_model.pkl'
PLOT_DIR = '../plots'  # Folder to save your presentation images

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Generates and saves a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Where did the model get confused?')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/rf_confusion_matrix.png')
    plt.close()
    print(f"   Saved confusion matrix to {PLOT_DIR}/rf_confusion_matrix.png")

def plot_feature_importance(model, feature_names):
    """Generates and saves a bar chart of the top 20 most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort descending
    
    # Take top 20
    top_n = 20
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Most Important Features (Random Forest)')
    plt.barh(range(top_n), importances[top_indices], align='center', color='green')
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis() # Highest importance on top
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/rf_feature_importance.png')
    plt.close()
    print(f"   Saved feature importance to {PLOT_DIR}/rf_feature_importance.png")

def main():
    print("1. Loading Master Data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} rows. Columns found: {list(df.columns)}")
    
    # -------------------------------------------------------
    # DATA SPLITTING (By Game ID)
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
    
    print(f"   Training on {len(train_game_ids)} games ({len(train_df)} moves)")
    print(f"   Testing on  {len(test_game_ids)} games ({len(test_df)} moves)")

    # -------------------------------------------------------
    # PREPARING FEATURES
    # -------------------------------------------------------
    drop_cols = ['game_id', 'result']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['result']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['result']
    
    # Encode labels: 0=Black, 1=Draw, 2=White
    print("   Encoding labels (0=Black, 1=Draw, 2=White)...")
    label_mapping = {0.0: 0, 0.5: 1, 1.0: 2}
    y_train = y_train.map(label_mapping).astype(int)
    y_test = y_test.map(label_mapping).astype(int)

    # -------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------
    print(f"3. Training Random Forest with config: {RF_CONFIG}...")
    
    rf_model = RandomForestClassifier(**RF_CONFIG)
    rf_model.fit(X_train, y_train)
    print("   Training Complete.")

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------
    print("4. Evaluating Model...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Black Win', 'Draw', 'White Win']))

    # -------------------------------------------------------
    # VISUALIZATIONS (The Presentation Stuff)
    # -------------------------------------------------------
    print("5. Generating Presentation Plots...")
    
    # Plot 1: Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, classes=['Black Win', 'Draw', 'White Win'])
    
    # Plot 2: Feature Importance
    # This proves if your new columns (Mobility) are actually doing anything!
    plot_feature_importance(rf_model, X_train.columns)

    # -------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------
    print(f"6. Saving model to {MODEL_PATH}")
    joblib.dump(rf_model, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()