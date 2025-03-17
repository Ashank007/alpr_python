from concurrent.futures import ThreadPoolExecutor
from superresolution import upscale_frame

executor = ThreadPoolExecutor(max_workers=2)  # Pool for enhancement threads

def enhance_plate_async(plate):
    """Runs plate enhancement asynchronously"""
    enhanced_plate = upscale_frame(plate)
    return enhanced_plate

def submit_plate_for_enhancement(plate):
    """Submit a plate for enhancement using a thread"""
    return executor.submit(enhance_plate_async, plate)  # ✅ Directly return the future


