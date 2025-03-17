import cv2
import time
from collections import Counter
import easyocr
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Initialize OCR reader globally
reader = easyocr.Reader(['en'], gpu=True)  

executor = ThreadPoolExecutor(max_workers=8)

# Constants
CONFIDENCE_THRESHOLD = 0.8
STABILIZATION_FRAMES = 10

# Store last detected texts per track ID
plate_memory = {}

def get_stable_text(track_id):
    """Returns the most stable OCR text using weighted voting."""
    if track_id not in plate_memory or not plate_memory[track_id]:
        return "UNKNOWN"
    
    text_counts = Counter(plate_memory[track_id])
    most_common_text, _ = text_counts.most_common(1)[0]
    return most_common_text

def enhance_plate(plate):
    if plate is None or plate.size == 0:
        return plate

    # Convert to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Adaptive Histogram Equalization (CLAHE) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Adaptive Thresholding for better text clarity
    thresholded = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return thresholded


def ocr_thread_function(plate, track_id):
    """Runs OCR in a separate thread."""
    global plate_memory  # Ensure we're using the global variable

    detected_texts = reader.readtext(plate, detail=1,allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",paragraph=False,decoder="wordbeamsearch")

    if detected_texts:
        ocr_results = [(text[1].strip().upper(), text[2]) for text in detected_texts]
        best_ocr_text, best_confidence = max(ocr_results, key=lambda x: x[1])

        if best_ocr_text and len(best_ocr_text) > 3 and best_confidence > CONFIDENCE_THRESHOLD:
            if track_id not in plate_memory:
                plate_memory[track_id] = []

            plate_memory[track_id].append(best_ocr_text)

            # Keep only the latest STABILIZATION_FRAMES number of texts
            if len(plate_memory[track_id]) > STABILIZATION_FRAMES:
                plate_memory[track_id].pop(0)


def print_detected_plates():
    """Continuously prints detected plates at an interval."""
    printed_plates = set()  # Store already printed plates

    while True:
        for track_id, texts in plate_memory.items():
            if texts:
                stable_plate = get_stable_text(track_id)
                if stable_plate and stable_plate not in printed_plates:
                    print(f"🚗 Detected License Plate: {stable_plate}")
                    printed_plates.add(stable_plate)

        time.sleep(1) 
