import os
import sys
import json
import wave
import struct
import math
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Constants
COUGH_DIR = os.path.join(os.path.dirname(__file__), '../datasets/cough')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/cough_model.h5')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '../models/cough_metrics.json')

# Mel Spectrogram Settings
SR = 16000
DURATION = 2.0  # seconds
N_MELS = 64
N_FTT = 1024
HOP_LENGTH = 512

def check_tensor_metal():
    """
    Checks if TensorFlow is using Apple Silicon Metal (GPU) acceleration.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow GPU Acceleration active: Found {len(gpus)} GPU(s).")
        for gpu in gpus:
            print(f" - Device: {gpu}")
        return True
    else:
        print("TensorFlow running on CPU mode.")
        return False

def generate_synthetic_data(num_samples=25):
    """
    Generates synthetic cough WAV files if no real data is found.
    """
    pos_dir = os.path.join(COUGH_DIR, 'positive')
    neg_dir = os.path.join(COUGH_DIR, 'negative')
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    print(f"Generating synthetic cough dataset: {num_samples} positive and {num_samples} negative samples.")
    
    # Generate positive samples (lower frequencies, rougher pitch, simulated cough explosions)
    for i in range(num_samples):
        filepath = os.path.join(pos_dir, f"cough_pos_{i:03d}.wav")
        if not os.path.exists(filepath):
            write_cough_wav(filepath, frequency=220, frequency_mod=100, is_positive=True)

    # Generate negative samples (higher frequency breathing, throat clearing)
    for i in range(num_samples):
        filepath = os.path.join(neg_dir, f"cough_neg_{i:03d}.wav")
        if not os.path.exists(filepath):
            write_cough_wav(filepath, frequency=440, frequency_mod=200, is_positive=False)

def write_cough_wav(filepath, frequency, frequency_mod, is_positive):
    """
    Writes a synthetic WAV file containing simulated cough signatures.
    """
    num_samples = int(DURATION * SR)
    amplitude = 32767
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit PCM
        wav_file.setframerate(SR)
        
        for i in range(num_samples):
            t = float(i) / SR
            
            # Simulate cough envelope (quick rise, rapid decay, multi-phase)
            if is_positive:
                # Double-exp cough phase envelope
                envelope = math.exp(-6.0 * t) * (1.0 - math.exp(-25.0 * t))
                # Add a second explosion phase
                if t > 0.4:
                    envelope += 0.5 * math.exp(-8.0 * (t - 0.4)) * (1.0 - math.exp(-20.0 * (t - 0.4)))
            else:
                # Regular single peak breathing/clearing envelope
                envelope = math.exp(-3.0 * t) * (1.0 - math.exp(-10.0 * t))
                
            # Synthesize signal (modulated sine + high frequency noise)
            freq_sweep = frequency + frequency_mod * math.sin(2 * math.pi * 3.0 * t)
            signal = math.sin(2.0 * math.pi * freq_sweep * t) * 0.6
            
            # Positive TB coughs have more turbulent noise (simulated wet cough)
            noise_amp = 0.4 if is_positive else 0.15
            noise = np.random.normal(0, 0.25) * noise_amp
            
            value = (signal + noise) * envelope
            value = max(-1.0, min(1.0, value))
            
            packed = struct.pack('<h', int(value * amplitude))
            wav_file.writeframesraw(packed)

def extract_spectrogram(filepath):
    """
    Loads a WAV file, extracts its Mel Spectrogram, and pads/crops it to a fixed size.
    """
    try:
        y, sr = librosa.load(filepath, sr=SR, duration=DURATION)
        # Pad if short
        target_len = int(SR * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
            
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FTT, hop_length=HOP_LENGTH
        )
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Norm to [0, 1]
        min_val = log_melspec.min()
        max_val = log_melspec.max()
        if max_val - min_val > 0:
            norm_melspec = (log_melspec - min_val) / (max_val - min_val)
        else:
            norm_melspec = np.zeros_like(log_melspec)
            
        return norm_melspec
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return None

def load_dataset():
    """
    Loads all WAV files from cough directory and extracts features.
    """
    # Verify dataset directories
    pos_dir = os.path.join(COUGH_DIR, 'positive')
    neg_dir = os.path.join(COUGH_DIR, 'negative')
    
    pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.wav')] if os.path.exists(pos_dir) else []
    neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.wav')] if os.path.exists(neg_dir) else []
    
    # Auto-generate synthetic if none exist
    if len(pos_files) == 0 and len(neg_files) == 0:
        generate_synthetic_data()
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.wav')]
        neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.wav')]
        
    features = []
    labels = []
    
    print("Extracting Mel Spectrograms...")
    for f in pos_files:
        spec = extract_spectrogram(f)
        if spec is not None:
            features.append(spec)
            labels.append(1)  # TB Positive
            
    for f in neg_files:
        spec = extract_spectrogram(f)
        if spec is not None:
            features.append(spec)
            labels.append(0)  # TB Negative
            
    X = np.array(features)
    y = np.array(labels)
    
    # Add channel dimension for CNN: (Samples, Height, Width, Channels)
    X = np.expand_dims(X, axis=-1)
    
    return X, y

def build_cnn_model(input_shape):
    """
    Builds a 2D CNN model for Mel Spectrogram classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_cough_model(epochs=10, batch_size=8, verbose=1):
    """
    Orchestrates dataset loading, splitting, model training, saving, and evaluation.
    """
    print("=" * 60)
    print(" Cough Audio CNN Training Pipeline Started ")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check GPU
    check_tensor_metal()
    
    # Load dataset
    X, y = load_dataset()
    print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
    
    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Model shape
    input_shape = X.shape[1:]  # (N_MELS, N_FRAMES, 1) e.g., (64, 63, 1)
    
    # Build Model
    model = build_cnn_model(input_shape)
    if verbose:
        model.summary()
        
    # Fit
    print(f"Training cough CNN model for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Save Model
    print(f"Saving cough model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    # Evaluate
    print("Evaluating model performance on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    y_test_flat = y_test.flatten()
    
    accuracy = float(accuracy_score(y_test_flat, y_pred))
    precision = float(precision_score(y_test_flat, y_pred, zero_division=0))
    recall = float(recall_score(y_test_flat, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_flat, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_test_flat, y_pred_probs))
    except Exception:
        roc_auc = 0.5
        
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    
    # Generate ROC curve coordinates for graphing
    try:
        fpr, tpr, _ = roc_curve(y_test_flat, y_pred_probs)
        roc_curve_data = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr, tpr)]
    except Exception:
        roc_curve_data = [{'fpr': 0.0, 'tpr': 0.0}, {'fpr': 1.0, 'tpr': 1.0}]
        
    # Generate confusion matrix
    cm = confusion_matrix(y_test_flat, y_pred)
    tn, fp, fn, tp = map(int, cm.ravel())
    
    # Log metrics to JSON
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        },
        'roc_curve': roc_curve_data,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        },
        'dataset_stats': {
            'total_samples': int(len(X)),
            'positive_samples': int(np.sum(y == 1)),
            'negative_samples': int(np.sum(y == 0))
        }
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved cough model metrics to {METRICS_PATH}")
    print("=" * 60)
    
if __name__ == '__main__':
    # Train 5 epochs if run directly for speed
    train_cough_model(epochs=5, batch_size=8)
