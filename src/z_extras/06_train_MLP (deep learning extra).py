import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# ==========================================
# 1. HYPERPARAMETER CONFIGURATION
# ==========================================
DL_CONFIG = {
    'layer_1_units': 512,
    'layer_2_units': 256,
    'layer_3_units': 128,
    'layer_4_units': 64,
    'dropout_rate': 0.2,     # Percentage of neurons to turn off (prevents memorization)
    'batch_size': 128,       # How many moves to learn from at once
    'epochs': 30,            # Max loops through the data
    'patience': 3            # Stop if no improvement after X epochs
}

# ==========================================
# 2. FILE PATHS
# ==========================================
INPUT_FILE = '../data/processed/master_data.csv'
MODEL_PATH = '../models/dl_model_optimized.keras' # Using .keras format
SCALER_PATH = '../models/scaler_dl.pkl'
PLOT_DIR = '../plots'

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_training_history(history):
    """
    Plots the Loss and Accuracy curves.
    This tells you if the model is learning or just memorizing.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Error)')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/dl_training_history.png')
    plt.close()
    print(f"   Saved training history to {PLOT_DIR}/dl_training_history.png")

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Neural Network (MLP)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/dl_confusion_matrix.png')
    plt.close()
    print(f"   Saved confusion matrix to {PLOT_DIR}/dl_confusion_matrix.png")

def main():
    print("1. Loading Master Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # -------------------------------------------------------
    # DATA PREP
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

    print("3. Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------
    # BUILDING THE BRAIN
    # -------------------------------------------------------
    print(f"4. Building Neural Network with config: {DL_CONFIG}...")
    
    model = Sequential()
    
    # Input Layer (Automatically adjusts to include Mobility columns)
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    
    # Layer 1
    model.add(Dense(DL_CONFIG['layer_1_units'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DL_CONFIG['dropout_rate']))
    
    # Layer 2
    model.add(Dense(DL_CONFIG['layer_2_units'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DL_CONFIG['dropout_rate']))
    
    # Layer 3
    model.add(Dense(DL_CONFIG['layer_3_units'], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DL_CONFIG['dropout_rate'] / 2)) # Lower dropout deeper in network
    
    # Layer 4
    model.add(Dense(DL_CONFIG['layer_4_units'], activation='relu'))
    model.add(Dropout(DL_CONFIG['dropout_rate'] / 2))
    
    # Output Layer
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()

    # -------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------
    print("5. Training...")
    
    stopper = EarlyStopping(monitor='val_loss', patience=DL_CONFIG['patience'], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=DL_CONFIG['epochs'],
        batch_size=DL_CONFIG['batch_size'],
        validation_split=0.1,
        callbacks=[stopper, reduce_lr],
        verbose=1
    )
    
    print("   Training Complete.")

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------
    print("6. Evaluating Model...")
    y_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Black Win', 'Draw', 'White Win']))

    # -------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------
    print("7. Generating Presentation Plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, classes=['Black Win', 'Draw', 'White Win'])

    # -------------------------------------------------------
    # SAVE
    # -------------------------------------------------------
    print("8. Saving model and scaler...")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("   Done.")

if __name__ == "__main__":
    main()