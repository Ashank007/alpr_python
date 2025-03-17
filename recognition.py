import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ocr_utils import get_stable_text,ocr_thread_function

executor = ThreadPoolExecutor(max_workers=6)

def recognize_plate(frame, tracked_objects):
    """Processes tracked plates and performs OCR."""
    ocr_tasks = 0
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        plate = frame[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        cv2.imshow(f"Cropped Plate {track_id}", plate)

        # Perform OCR in a separate threa
        executor.submit(ocr_thread_function, plate, track_id)

        final_plate_text = get_stable_text(track_id)
        ocr_tasks += 1
        # Draw bounding box and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}: {final_plate_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)



