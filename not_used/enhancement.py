from concurrent.futures import ThreadPoolExecutor
from process_enhanced_plate import submit_plate_for_enhancement

executor = ThreadPoolExecutor(max_workers=2)
enhanced_plates = {}  # Stores enhancement results

def enhance_plate(plate, track_id, frame_count):
    """
    Submits the plate for enhancement every 60 frames (~2 seconds at 30 FPS).
    Stores the future result for later retrieval.
    """
    if frame_count % 60 == 0:
        future = submit_plate_for_enhancement(plate)  # ✅ No need to wrap in executor.submit()
        enhanced_plates[track_id] = future  # Store future to track processing completion

def get_enhanced_plate(track_id):
    """
    Retrieves the enhanced plate once processing is complete.
    Returns the enhanced plate if available, otherwise None.
    """
    future = enhanced_plates.get(track_id)
    if future and future.done():
        return future.result()
    return None


