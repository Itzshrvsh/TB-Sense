# TB-Sense AI: Multimodal Tuberculosis Prediction Dashboard

TB-Sense AI is a state-of-the-art, on-device diagnostic screening dashboard designed for macOS. It integrates deep learning algorithms and a rule-based clinical scoring engine to screen for Tuberculosis (TB) using three modalities:
1. **Cough Audio Analysis**: A 2D Convolutional Neural Network trained on log-mel spectrogram features extracted from raw WAV audio clips.
2. **Chest X-Ray Imaging**: Transfer learning and classification using an `EfficientNetB0` CNN.
3. **Clinical Risk Assessment**: An 11-point clinical symptom rule-engine based on World Health Organization (WHO) screening markers.

The app is fully responsive, runs completely offline locally on-device, and utilizes **Apple Silicon GPU acceleration** (TensorFlow Metal) when available.

---

## 🛠️ Apple Silicon Setup & Installation

Follow these steps to configure your Python environment on a macOS machine:

### 1. Prerequisites
- **Operating System**: macOS 12.0+ (Monterey or later recommended).
- **Processor**: Apple Silicon (M1, M2, M3, M4 series chip).
- **Python**: Version `3.10` or `3.11` is highly recommended for compatibility with TensorFlow Metal wheels.

### 2. Configure Virtual Environment
Create a clean virtual environment in the project directory:
```bash
# Clone or navigate to the workspace
cd tb_dashboard

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
Install the required libraries. The `requirements.txt` file uses PEP-508 markers to automatically install `tensorflow-macos` and `tensorflow-metal` when running on macOS ARM64 architectures:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Dashboard

Start the Flask application server:
```bash
python3 app.py
```
By default, the server runs on **`http://127.0.0.1:5001`**. Open this URL in your web browser.

### 🤖 Automatic Training System on Startup
1. On start, the server checks if trained model checkpoints exist in `models/cough_model.h5` and `models/xray_model.h5`.
2. **Missing Checkpoints**: The app automatically launches a background training thread to compile both models.
3. **Dataset Detection & Fallback**:
   - The engine searches for files under `datasets/cough/` and `datasets/xray/`.
   - If empty, the app **automatically generates synthetic WAV and PNG files** representing clinical TB patterns.
   - It then preprocesses the data (Mel spectrograms for audio, EfficientNet normalization for images) and trains the models in under 45 seconds.
4. **Live Training Page**: Navigate to the **Training Status** page in the dashboard to review the compiler logs in real-time.

---

## 🖥️ Dashboard Page Highlights

- **Dashboard Page**: Displays high-level triage volumes, hardware acceleration status (e.g., Apple Silicon GPU Active), and modality weight summaries.
- **Diagnostic Portal**:
  - Drag-and-drop or select files for **Cough Audio** (.wav, .mp3) and **Chest X-Ray** (.png, .jpg).
  - Fill out patient demographics and symptoms (fever, hemoptysis, chronic cough, smoking history).
  - Submit to calculate individual and combined **multimodal late fusion classification scores** (`0.5 * X-ray + 0.3 * Cough + 0.2 * Clinical`).
  - Displays side-by-side chest X-rays with a **Grad-CAM** heat map highlighting lung activation sectors.
  - Automatically compiles and exports a clean, print-ready PDF health report.
- **Model Metrics**: Reviews quantitative model accuracy, F1-scores, true positive rates, dynamically drawn SVG ROC curves, and detailed confusion matrices.
- **Dataset Statistics**: Visualizes database label counts and imbalances.

---

## 🔬 Model Technical Specs

### Cough Spectrogram 2D CNN
- **Inputs**: Mel Spectrograms (64 Mel bins, 16000Hz sampling rate, 2-second clip length) mapped to a `(64, 63, 1)` tensor.
- **Architecture**: 3 Convolutional layers (32, 64, 128 channels) with Batch Normalization, Max Pooling, Dropout (0.25 - 0.40), and Dense layers.
- **Outputs**: Sigmoid output probability [0, 1].

### Chest X-Ray EfficientNetB0
- **Inputs**: 3-channel RGB image tensor sized `(224, 224, 3)`.
- **Architecture**: ImageNet pre-trained `EfficientNetB0` backbone acting as a feature extractor, routing output to Global Average Pooling, Dropout (0.4), and a Dense classification head.
- **Grad-CAM Implementation**: Computes the gradients of the classification score with respect to the `top_activation` layer in the EfficientNet backbone to produce activation heatmaps.
