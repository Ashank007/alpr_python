
import torch
import cv2
import numpy as np
from realesrgan import RealESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_frame(frame, scale=4):
    """
    Upscales a given image frame using Real-ESRGAN.

    Args:
        frame (numpy.ndarray): Input image frame (BGR format).
        scale (int): The upscaling factor (default is 4).

    Returns:
        numpy.ndarray: The upscaled image frame (BGR format).
    """
    if frame is None:
        raise ValueError("Input frame is None")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Real-ESRGAN model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGAN(device, model, scale=scale)

    # Load pre-trained weights
    upsampler.load_weights("RealESRGAN_x4plus.pth")  # Ensure this file is in the correct directory

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform super-resolution
    try:
        upscaled_rgb = upsampler.enhance(frame_rgb)
        upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
        return upscaled_bgr
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return frame  # Return original frame if upscaling fails

# Example usage:
# frame = cv2.imread("input.jpg")  # Load an image
# upscaled_frame = upscale_frame(frame)
# cv2.imshow("Upscaled Frame", upscaled_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


