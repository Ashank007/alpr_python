import onnxruntime as ort
import numpy as np
import cv2
import os

def upscale_image(model_path, input_image_path, max_dim=1024):
    """Upscales an image using RealESRGAN ONNX model and displays comparison."""
    
    # ✅ Enable ONNX memory optimizations
    os.environ["ORT_NO_CUDA_GRAPH"] = "1"
    os.environ["ORT_DISABLE_MEMORY_ARENA"] = "1"

    # ✅ Load ONNX model with GPU if available, otherwise CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    print(f"Using ONNX Runtime with: {providers[0]}")

    ort_session = ort.InferenceSession(model_path, providers=providers)

    # ✅ Read input image
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ✅ Resize input if too large (Prevents out-of-memory errors)
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)  # Calculate scale factor

    if scale < 1.0:  # Only resize if larger than max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image to: {new_w}x{new_h}")

    # ✅ Normalize and reshape for model input
    image_float = image.astype(np.float32) / 255.0  # Scale to [0,1]
    image_float = np.transpose(image_float, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
    image_float = np.expand_dims(image_float, axis=0)  # Add batch dimension (1, C, H, W)

    # ✅ Get correct ONNX input name
    input_name = ort_session.get_inputs()[0].name  

    # ✅ Run inference (Upscale the image)
    print("Running ONNX inference...")
    outputs = ort_session.run(None, {input_name: image_float})

    # ✅ Convert output image back to original format
    output = np.squeeze(outputs[0])  # Remove batch dimension
    output = np.transpose(output, (1, 2, 0))  # Convert (C, H, W) → (H, W, C)
    output = (output * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
    
    # ✅ Resize original image to match upscaled dimensions
    h, w = output.shape[:2]
    image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    # ✅ Convert images to BGR before displaying
    image_resized_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # ✅ Stack images side by side for comparison
    comparison = np.hstack((image_resized_bgr, output_bgr))

    # ✅ Show comparison
    cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("Comparison", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
upscale_image("models/RealESRGAN_x4plus.onnx", "1.webp")


