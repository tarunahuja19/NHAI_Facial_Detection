"""
model.py — Face Detection Model
NHAI Hackathon · Redesigned MBConv-based CNN

3 output heads:
  1. box_output        → [cx, cy, w, h]         4 numbers
  2. landmark_output   → [x,y] × 5 points       10 numbers
  3. confidence_output → face present? (0 or 1)  1 number
"""

import tensorflow as tf
from keras import layers, Model


def mbconv_block(x, in_channels, out_channels, expansion_factor=3, kernel_size=3, stride=1, use_se=True):
    """
    MobileNetV3-style Inverted Residual Block (MBConv).
    
    Includes:
      - 1x1 expansion convolution (if expansion_factor > 1)
      - Depthwise convolution (strided if stride > 1)
      - Squeeze-and-Excitation (SE) block for channel attention
      - 1x1 projection convolution back to target channel size
      - Residual connection (if stride == 1 and in_channels == out_channels)
    """
    residual = x
    hidden_dim = in_channels * expansion_factor

    # ── 1. EXPANSION (1x1 Conv) ───────────────────────────────────────────────
    if expansion_factor != 1:
        x = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)

    # ── 2. DEPTHWISE CONVOLUTION ──────────────────────────────────────────────
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    # ── 3. SQUEEZE-AND-EXCITATION (SE) ────────────────────────────────────────
    if use_se:
        se = layers.GlobalAveragePooling2D()(x)
        se_channels = max(1, hidden_dim // 4)
        se = layers.Dense(se_channels, activation='relu', use_bias=True)(se)
        se = layers.Dense(hidden_dim, activation='sigmoid', use_bias=True)(se)
        # Reshape to (1, 1, hidden_dim) for element-wise broadcast multiplication
        se = layers.Reshape((1, 1, hidden_dim))(se)
        x = layers.Multiply()([x, se])

    # ── 4. PROJECTION (1x1 Conv) ──────────────────────────────────────────────
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # ── 5. RESIDUAL CONNECTION ───────────────────────────────────────────────
    if stride == 1 and in_channels == out_channels:
        x = layers.Add()([residual, x])

    return x


def build_face_detection_model(input_size=128):
    """
    Builds the complete redesigned face detection model using MBConv blocks.

    Input shape:  (batch, 128, 128, 3)
    Output:
      box_output        → (batch, 4)   — [cx, cy, w, h], all in [0, 1]
      landmark_output   → (batch, 10)  — [lx,ly, rx,ry, nx,ny, lmx,lmy, rmx,rmy], all in [0, 1]
      confidence_output → (batch, 1)   — probability face is present, in [0, 1]
    """
    inputs = layers.Input(shape=(input_size, input_size, 3), name="image_input")

    # ── BACKBONE ──────────────────────────────────────────────────────────────
    # Initial standard Conv stem: 128x128 -> 64x64
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    # Stage 1: 64x64, channels: 16 -> 24
    x = mbconv_block(x, in_channels=16, out_channels=24, expansion_factor=3, kernel_size=3, stride=1, use_se=True)

    # Stage 2: 64x64 -> 32x32, channels: 24 -> 32
    x = mbconv_block(x, in_channels=24, out_channels=32, expansion_factor=3, kernel_size=3, stride=2, use_se=True)

    # Stage 3: 32x32, channels: 32 -> 48
    x = mbconv_block(x, in_channels=32, out_channels=48, expansion_factor=3, kernel_size=3, stride=1, use_se=True)

    # Stage 4: 32x32 -> 16x16, channels: 48 -> 64
    x = mbconv_block(x, in_channels=48, out_channels=64, expansion_factor=3, kernel_size=3, stride=2, use_se=True)

    # Stage 5: 16x16, channels: 64 -> 96
    x = mbconv_block(x, in_channels=64, out_channels=96, expansion_factor=3, kernel_size=3, stride=1, use_se=True)

    # Stage 6: 16x16 -> 8x8, channels: 96 -> 128 (uses 5x5 kernel for larger receptive field)
    x = mbconv_block(x, in_channels=96, out_channels=128, expansion_factor=3, kernel_size=5, stride=2, use_se=True)

    # Stage 7: 8x8, channels: 128 -> 128 (uses 5x5 kernel for larger receptive field)
    x = mbconv_block(x, in_channels=128, out_channels=128, expansion_factor=3, kernel_size=5, stride=1, use_se=True)

    # ── POOLING ───────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    # Shape: 128 channels

    # ── SHARED DENSE LAYER ────────────────────────────────────────────────────
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # ── OUTPUT HEADS ──────────────────────────────────────────────────────────
    box_output = layers.Dense(4, activation='sigmoid', name="box_output")(x)
    landmark_output = layers.Dense(10, activation='sigmoid', name="landmark_output")(x)
    confidence_output = layers.Dense(1, activation='sigmoid', name="confidence_output")(x)

    model = Model(
        inputs=inputs,
        outputs=[box_output, landmark_output, confidence_output],
        name="face_detector"
    )
    return model


if __name__ == "__main__":
    # If tensorflow can be imported, print model summary
    try:
        model = build_face_detection_model()
        model.summary()

        total_params = model.count_params()
        print(f"\nTotal parameters:          {total_params:,}")
        print(f"Size before quantization:  {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"Expected after INT8 quant: ~{total_params * 1 / 1024 / 1024:.2f} MB")
        print("\nOutput shapes:")
        for output in model.outputs:
            print(f"  {output.name}: {output.shape}")
    except ImportError:
        print("TensorFlow/Keras is not installed. To run model summary, install TensorFlow.")
