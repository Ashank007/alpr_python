import sys
import numpy as np
from config import SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESHOLD

sys.path.append("./sort")
from sort import Sort  # Import SORT tracker

def initialize_tracker():
    """Initializes the SORT tracker."""
    return Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)

def track_objects(tracker, detections):
    """Updates the tracker with new detections."""
    return tracker.update(detections)

