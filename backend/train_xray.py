import os
import sys
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Constants
XRAY_DIR = os.path.join(os.path.dirname(__file__), '../datasets/xray')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xray_model.h5')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '../models/xray_metrics.json')
IMG_SIZE = 224

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

def generate_synthetic_xrays(num_samples=25):
    """
    Generates synthetic chest X-ray PNG files if no real data is found.
    Creates mock chest cavities, lungs, and consolidations for TB positive cases.
    """
    tb_dir = os.path.join(XRAY_DIR, 'tuberculosis')
    normal_dir = os.path.join(XRAY_DIR, 'normal')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    print(f"Generating synthetic X-ray dataset: {num_samples} TB and {num_samples} Normal samples.")

    for i in range(num_samples):
        # Normal Chest X-ray
        normal_path = os.path.join(normal_dir, f"xray_normal_{i:03d}.png")
        if not os.path.exists(normal_path):
            img = draw_chest_xray(has_tb=False)
            cv2.imwrite(normal_path, img)
            
        # TB Chest X-ray
        tb_path = os.path.join(tb_dir, f"xray_tb_{i:03d}.png")
        if not os.path.exists(tb_path):
            img = draw_chest_xray(has_tb=True)
            cv2.imwrite(tb_path, img)

def draw_chest_xray(has_tb):
    """
    Draws a synthetic gray chest X-ray with spinal cord, ribcage shadows, 
    and lung structures, adding lesions/consolidations if has_tb=True.
    """
    # Create black canvas
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8) + 20
    
    # 1. Draw chest cavity contour (semi-elliptical rib structure)
    cv2.ellipse(img, (IMG_SIZE // 2, IMG_SIZE // 2 + 10), (90, 80), 0, 0, 360, 50, -1)
    
    # 2. Draw left and right lung lobes (darker regions inside the chest)
    # Left Lung
    cv2.ellipse(img, (IMG_SIZE // 2 - 35, IMG_SIZE // 2 + 5), (28, 60), 0, 0, 360, 15, -1)
    # Right Lung
    cv2.ellipse(img, (IMG_SIZE // 2 + 35, IMG_SIZE // 2 + 5), (28, 60), 0, 0, 360, 15, -1)
    
    # 3. Draw spinal column shadow (vertical bar in center)
    cv2.rectangle(img, (IMG_SIZE // 2 - 5, 20), (IMG_SIZE // 2 + 5, IMG_SIZE - 20), 120, -1)
    
    # 4. Add rib cages (horizontal arches)
    for y in range(40, IMG_SIZE - 40, 20):
        # Left ribs
        cv2.ellipse(img, (IMG_SIZE // 2 - 100, y + 10), (90, 15), 0, 320, 360, 75, 2)
        # Right ribs
        cv2.ellipse(img, (IMG_SIZE // 2 + 100, y + 10), (90, 15), 0, 180, 220, 75, 2)
        
    # 5. Add clavicles (collar bones)
    cv2.line(img, (30, 45), (IMG_SIZE // 2 - 10, 55), 130, 6)
    cv2.line(img, (IMG_SIZE - 30, 45), (IMG_SIZE // 2 + 10, 55), 130, 6)
    
    # 6. Apply gaussian blur to make it look organic like an X-ray film
    img = cv2.GaussianBlur(img, (9, 9), 0)
    
    # 7. Add TB-specific signs (white patchy infiltrates / consolidations in lungs)
    if has_tb:
        # Create an overlay for infiltrates
        overlay = img.copy()
        # Randomly choose left or right upper lung field (classic TB cavity location)
        side = np.random.choice([-1, 1])
        center = (IMG_SIZE // 2 + side * 35, IMG_SIZE // 2 - 25)
        # Draw patchy lesions
        cv2.circle(overlay, center, 14, 180, -1)
        cv2.circle(overlay, (center[0] + 8, center[1] + 8), 10, 160, -1)
        cv2.circle(overlay, (center[0] - 8, center[1] + 4), 8, 140, -1)
        # Blur the consolidations
        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        # Blend overlay
        img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
        
    # Add random scanner noise
    noise = np.random.normal(0, 3.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return img

def load_dataset():
    """
    Loads all chest X-ray images, resizes them, converts to 3-channel RGB, 
    and normalizes pixel values.
    """
    tb_dir = os.path.join(XRAY_DIR, 'tuberculosis')
    normal_dir = os.path.join(XRAY_DIR, 'normal')
    
    tb_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(tb_dir) else []
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(normal_dir) else []
    
    # Auto-generate synthetic if empty
    if len(tb_files) == 0 and len(normal_files) == 0:
        generate_synthetic_xrays()
        tb_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    images = []
    labels = []
    
    print("Loading and preprocessing images...")
    for f in tb_files:
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Convert to RGB (EfficientNet requirement)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(1)  # TB Positive
            
    for f in normal_files:
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(0)  # TB Negative
            
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    return X, y

def build_transfer_model(input_shape):
    """
    Builds the Chest X-ray classifier using pre-trained EfficientNetB0 backbone.
    """
    # Load EfficientNetB0 backbone (frozen weights, include_top=False)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze transfer layers for standard execution
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_xray_model(epochs=10, batch_size=8, verbose=1):
    """
    Orchestrates X-ray dataset loading, splitting, EfficientNet training, and metrics logging.
    """
    print("=" * 60)
    print(" Chest X-Ray CNN Training Pipeline Started ")
    print("=" * 60)
    
    # Ensure directory structure
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Check GPU
    check_tensor_metal()
    
    # Load dataset
    X, y = load_dataset()
    print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"TB Positive: {np.sum(y == 1)}, Normal: {np.sum(y == 0)}")
    
    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Model
    model = build_transfer_model((IMG_SIZE, IMG_SIZE, 3))
    if verbose:
        model.summary()
        
    # Fit
    print(f"Training chest X-ray model for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Save Model
    print(f"Saving chest X-ray model to {MODEL_PATH}...")
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
        
    print(f"Saved chest X-ray model metrics to {METRICS_PATH}")
    print("=" * 60)

if __name__ == '__main__':
    # Train 5 epochs if run directly for speed
    train_xray_model(epochs=5, batch_size=8)
