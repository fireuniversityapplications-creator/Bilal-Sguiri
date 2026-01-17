import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ==========================================
# 1. HYPERPARAMETER CONFIGURATION
# ==========================================
# This is the "Basic" config (low neighbors, no PCA)
KNN_BASIC_CONFIG = {
    'n_neighbors': 5,          # Default is 5 (Low = Sensitive to noise)
    'metric': 'minkowski',     # Default metric (Euclidean distance)
    'n_jobs': -1               # Use all CPUs
}

# We still need to limit the size, or standard KNN will crash/freeze
TRAINING_SAMPLE_SIZE = 50000 

# ==========================================
# 2. FILE PATHS
# ==========================================
INPUT_FILE = '../data/processed/master_data.csv'
MODEL_PATH = '../models/knn_model_basic.pkl'  # Saved as "basic"
SCALER_PATH = '../models/scaler_knn_basic.pkl'
PLOT_DIR = '../plots'

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # Using a different color (Greys) to distinguish from the "Good" KNN (Oranges)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: KNN (Basic / No PCA)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    # Save with a unique name so we don't overwrite the good one!
    plt.savefig(f'{PLOT_DIR}/knn_basic_confusion_matrix.png')
    plt.close()
    print(f"   Saved confusion matrix to {PLOT_DIR}/knn_basic_confusion_matrix.png")

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
    
    # -------------------------------------------------------
    # DOWNSAMPLING
    # -------------------------------------------------------
    print(f"   Original Training Size: {len(train_df)}")
    print(f"   Reducing to {TRAINING_SAMPLE_SIZE} for KNN speed...")
    
    if len(train_df) > TRAINING_SAMPLE_SIZE:
        train_df = train_df.sample(n=TRAINING_SAMPLE_SIZE, random_state=42)
    
    print(f"   Final Training Size: {len(train_df)}")

    drop_cols = ['game_id', 'result']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['result']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['result']

    label_mapping = {0.0: 0, 0.5: 1, 1.0: 2}
    y_train = y_train.map(label_mapping).astype(int)
    y_test = y_test.map(label_mapping).astype(int)

    # -------------------------------------------------------
    # SCALING (Mandatory for KNN)
    # -------------------------------------------------------
    print("3. Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform full test set
    X_test_scaled = scaler.transform(X_test)
    print("   Scaling Complete.")

    # -------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------
    print(f"4. Training Basic KNN with config: {KNN_BASIC_CONFIG}...")
    
    knn = KNeighborsClassifier(**KNN_BASIC_CONFIG)
    knn.fit(X_train_scaled, y_train)
    print("   Training Complete.")

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------
    print("5. Evaluating Model (on 10,000 test samples)...")
    
    # Evaluating on a subset for speed
    X_test_subset = X_test_scaled[0:10000]
    y_test_subset = y_test.iloc[0:10000]
    
    y_pred = knn.predict(X_test_subset)
    
    acc = accuracy_score(y_test_subset, y_pred)
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test_subset, y_pred, target_names=['Black Win', 'Draw', 'White Win']))

    # -------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------
    print("6. Generating Presentation Plots...")
    plot_confusion_matrix(y_test_subset, y_pred, classes=['Black Win', 'Draw', 'White Win'])

    # -------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------
    print(f"7. Saving model...")
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("   Done.")

if __name__ == "__main__":
    main()