# 🧠 NHAI Model Training & Optimization Pipeline (`model-training`)

This directory houses the end-to-end Machine Learning pipelines for the **NHAI Biometric Attendance & Liveness Verification System**. It contains the models, training configurations, validation scripts, and quantization utilities for both the Face Detector and the Face Recognition/Liveness backbones.

---

## 🧭 Pipeline Overview

The biometric verification system relies on two distinct models executing locally on the client's device:

1. **Face Detector (TensorFlow/Keras)**: A super-lightweight MBConv-based model that detects the face bounding box, identifies 5 key landmarks (eyes, nose, mouth corners), and calculates the face presence confidence score.
2. **Face Recognition & Liveness Backbone (JAX/Flax)**: The `Iris` network that maps the cropped face to a 512-dimensional embedding and determines liveness features under a strict **4 MB INT8 memory budget**.

```mermaid
graph TD
    subgraph Face Detection (TF/Keras)
        det_src[model_detector.py] -->|Verify Shapes| det_verify[verify_model.py]
        det_src -->|INT8 Quantize| det_quant[quantize_detector.py]
        det_quant -->|Export| det_onnx[detector.onnx]
    end
    
    subgraph Face Recognition & Liveness (JAX/Flax)
        rec_src[model.py] -->|Losses: AdaFace/ArcFace| rec_loss[losses.py]
        rec_src -->|Train Scripts| rec_train[train.py]
        rec_train -->|Evaluate LFW| rec_eval[evaluate_lfw.py]
        rec_train -->|Export ONNX| rec_export[export_onnx.py]
        rec_export -->|INT8 Quantize| rec_quant[quantize_onnx.py]
        rec_quant -->|Export| rec_onnx[face_model_quant.onnx]
    end
    
    det_onnx -->|Copy to Assets| mobile_assets[artifacts/mobile/assets/]
    rec_onnx -->|Copy to Assets| mobile_assets
```

---

## 📂 File Breakdown & Responsibilities

### 1. Face Detection Model

* **[`model_detector.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/model_detector.py)**: Definess the MBConv-based (MobileNetV3-style Inverted Residual Block) convolutional neural network. Features Squeeze-and-Excitation (SE) channel attention, and outputs 3 heads:
  1. `box_output` (`[cx, cy, w, h]`)
  2. `landmark_output` (`[lx, ly, rx, ry, nx, ny, lmx, lmy, rmx, rmy]`)
  3. `confidence_output` (probability that a face is present)
* **[`verify_model.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/verify_model.py)**: Validation script utilizing a custom Mock Keras layer parser to verify shape propagation and calculate total parameters without needing a complete TensorFlow installation.
* **[`param_calc.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/param_calc.py)** & **[`param_calc_v2.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/param_calc_v2.py)**: Performance inspection scripts evaluating computational latency, FLOP counts, and parameter allocations for different model variants.
* **[`quantize_detector.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/quantize_detector.py)**: Utilities to post-quantize the Keras weights into an optimized INT8 model format.

### 2. Face Recognition & Liveness (`Iris`)

* **[`model.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/model.py)**: JAX/Flax implementation of the production-ready `Iris` architecture. Specifically engineered with structural re-parameterization token mixers (`RepMix`), Lite Multi-Head Linear Attention (`LiteMHLA`), and Learned Graph Attention (`LearnedGAT`) to achieve highly accurate 512-dim embedding representation within a sub-4MB footprint.
* **[`losses.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/losses.py)**: Numerical-stable implementations of advanced training loss layers:
  * **AdaFace**: An adaptive margin loss matching image quality (essential for low-illumination and outdoor toll plaza settings).
  * **ArcFace**: Standard additive angular margin loss.
* **[`train.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/train.py)**: The main multi-device training runner. Integrates HuggingFace `datasets` streaming loaders, JAX GSPMD 1D data-parallel sharding, and Orbax checkpoint manager.
* **[`train_casia_fixed.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/train_casia_fixed.py)** / **[`train_digiface_cluster.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/train_digiface_cluster.py)** / **[`train_vggface2_cluster.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/train_vggface2_cluster.py)**: Specialized cluster configurations and Hyperparameter adjustments for different training sets (CASIA-WebFace, DigiFace-1M, VGGFace2).
* **[`evaluate_lfw.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/evaluate_lfw.py)**: Validates recognition accuracy and cosine similarity thresholds against the Labeled Faces in the Wild (LFW) benchmark.
* **[`export_onnx.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/export_onnx.py)**: Compiles the Flax module variables, strips out auxiliary loss layers, and exports the pure backbone graph into ONNX format.
* **[`quantize_onnx.py`](file:///home/jemin/Desktop/Code/nhai-app/model-training/quantize_onnx.py)**: Applies Post-Training Quantization (PTQ) to convert weight tensors to INT8, shrinking the model size to `~3.89 MB` and outputting the final `face_model_quant.onnx`.

---

## 🚀 Getting Started

### 📋 Prerequisites

Install the Python machine learning stack:
```bash
pip install jax flax optax tensorflow keras onnx onnxruntime tf2onnx opencv-python datasets tqdm numpy pillow orbax-checkpoint
```

### 🏋️ Training the Recognition Model

To train the `Iris` backbone on a multi-GPU node using VGGFace2:
```bash
python train.py \
    --dataset_name "chronopt-research/cropped-vggface2-224" \
    --loss_type "adaface" \
    --batch_size 128 \
    --num_epochs 25 \
    --base_lr 0.03
```

### 🧪 Verifying the Face Detector Model

Run the fast shapes and parameters check:
```bash
python verify_model.py
```

### 📦 Exporting & Quantizing to ONNX

Once training is complete, export and quantize your weights to make them compatible with `onnxruntime-react-native`:

1. **Export the Flax model**:
   ```bash
   python export_onnx.py --checkpoint_path "/path/to/checkpoint" --output_path "./backbone.onnx"
   ```
2. **Quantize the exported model**:
   ```bash
   python quantize_onnx.py --input_path "./backbone.onnx" --output_path "./face_model_quant.onnx"
   ```
3. **Deploy**:
   Copy the final `face_model_quant.onnx` directly into your [mobile assets folder](file:///home/jemin/Desktop/Code/nhai-app/artifacts/mobile/assets).
