import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ==========================================
# 1. HYPERPARAMETER CONFIGURATION
# ==========================================
KNN_CONFIG = {
    'n_neighbors': 25,         # Higher = Smoother decision boundary
    'metric': 'manhattan',     # Distance metric (Manhattan often better for high dim)
    'n_jobs': -1
}

TRAINING_SAMPLE_SIZE = 100000  # Keep this to prevent crashing
PCA_COMPONENTS = 20            # Number of features to keep after compression

# ==========================================
# 2. FILE PATHS
# ==========================================
INPUT_FILE = '../data/processed/master_data.csv'
MODEL_PATH = '../models/knn_model_optimized.pkl'
SCALER_PATH = '../models/scaler_knn.pkl'
PCA_PATH = '../models/pca_knn.pkl'
PLOT_DIR = '../plots'

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: KNN (Optimized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/knn_confusion_matrix.png')
    plt.close()
    print(f"   Saved confusion matrix to {PLOT_DIR}/knn_confusion_matrix.png")

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
    print(f"   Reducing training data to {TRAINING_SAMPLE_SIZE} rows...")
    if len(train_df) > TRAINING_SAMPLE_SIZE:
        train_df = train_df.sample(n=TRAINING_SAMPLE_SIZE, random_state=42)

    drop_cols = ['game_id', 'result']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['result']
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['result']

    label_mapping = {0.0: 0, 0.5: 1, 1.0: 2}
    y_train = y_train.map(label_mapping).astype(int)
    y_test = y_test.map(label_mapping).astype(int)

    # -------------------------------------------------------
    # PIPELINE: SCALING -> PCA
    # -------------------------------------------------------
    print("3. Applying Scaling and PCA...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"   Data shape compressed to {X_train_pca.shape}")

    # -------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------
    print(f"4. Training KNN with config: {KNN_CONFIG}...")
    
    knn = KNeighborsClassifier(**KNN_CONFIG)
    knn.fit(X_train_pca, y_train)
    print("   Training Complete.")

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------
    print("5. Evaluating Model (on 20,000 test samples)...")
    
    X_test_subset = X_test_pca[0:20000]
    y_test_subset = y_test.iloc[0:20000]
    
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
    # SAVE
    # -------------------------------------------------------
    print(f"7. Saving models...")
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    print("   Done.")

if __name__ == "__main__":
    main()