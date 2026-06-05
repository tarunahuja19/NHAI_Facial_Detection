import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    model_fp32 = "./scratch/face_model.onnx"
    model_quant = "./scratch/face_model_quant.onnx"
    
    print("Quantizing ONNX model to INT8...")
    # Quantize the model
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        weight_type=QuantType.QInt8
    )
    print("Quantization completed!")
    
    # Check file sizes
    size_fp32 = os.path.getsize(model_fp32) / (1024 * 1024)
    size_quant = os.path.getsize(model_quant) / (1024 * 1024)
    print(f"FP32 Model Size: {size_fp32:.2f} MB")
    print(f"INT8 Model Size: {size_quant:.2f} MB")
    
    # Verification: Run inference on both models with random input and compare cosine similarity
    print("\nVerifying model output consistency...")
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
    
    # Run FP32 model
    sess_fp32 = ort.InferenceSession(model_fp32)
    outputs_fp32 = sess_fp32.run(None, {"input_image": test_input})
    emb_fp32 = outputs_fp32[0][0]
    
    # Run INT8 model
    sess_quant = ort.InferenceSession(model_quant)
    outputs_quant = sess_quant.run(None, {"input_image": test_input})
    emb_quant = outputs_quant[0][0]
    
    # Compute cosine similarity
    dot_product = np.dot(emb_fp32, emb_quant)
    norm_fp32 = np.linalg.norm(emb_fp32)
    norm_quant = np.linalg.norm(emb_quant)
    similarity = dot_product / (norm_fp32 * norm_quant)
    
    print(f"Embedding Dimensions: {len(emb_fp32)}")
    print(f"Cosine Similarity between FP32 and INT8 embeddings: {similarity:.6f}")
    
    if similarity >= 0.95:
        print("Success: Quantized model output is highly aligned with original model!")
    else:
        print("Warning: Cosine similarity is low. Quantization might have introduced errors.")

if __name__ == "__main__":
    main()
