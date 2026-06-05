import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    model_fp32 = "./scratch/version-slim-320.onnx"
    model_quant = "./scratch/version-slim-320_quant.onnx"
    
    print("Quantizing face detector ONNX model to INT8...")
    # Quantize the model
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        weight_type=QuantType.QInt8
    )
    print("Quantization completed!")
    
    size_fp32 = os.path.getsize(model_fp32) / (1024 * 1024)
    size_quant = os.path.getsize(model_quant) / (1024 * 1024)
    print(f"FP32 Model Size: {size_fp32:.2f} MB")
    print(f"INT8 Model Size: {size_quant:.2f} MB")
    
    # Run a dummy inference to inspect the scores range
    sess = ort.InferenceSession(model_quant)
    test_input = np.random.randn(1, 3, 240, 320).astype(np.float32)
    outputs = sess.run(None, {"input": test_input})
    
    scores = outputs[0] # [1, 4420, 2]
    boxes = outputs[1]  # [1, 4420, 4]
    
    print("\nInference Output Inspection:")
    print("Scores shape:", scores.shape)
    print("Boxes shape:", boxes.shape)
    
    # Check if scores sum to 1.0 (indicating softmax is in the graph)
    sample_scores = scores[0, :5, :]
    print("Sample scores:")
    print(sample_scores)
    
    row_sums = np.sum(sample_scores, axis=1)
    print("Sample scores row sums:", row_sums)
    is_softmax = np.allclose(row_sums, 1.0, atol=1e-3)
    print("Softmax already applied in graph?", is_softmax)

if __name__ == "__main__":
    main()
