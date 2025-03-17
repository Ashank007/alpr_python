import cv2
import time
import threading
from detection import load_model, detect_plates
from tracking import initialize_tracker, track_objects
from recognition import recognize_plate 
from utils import calculate_fps
from ocr_utils import print_detected_plates

def main(video_path):
    model = load_model()
    tracker = initialize_tracker()
    cap = cv2.VideoCapture(video_path)
    plate_printing_thread = threading.Thread(target=print_detected_plates, daemon=True)
    plate_printing_thread.start()
    frame_count = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_plates(model, frame)
        tracked_objects = track_objects(tracker, detections)
        
        recognize_plate(frame, tracked_objects)

        avg_fps = calculate_fps(start_time)
    
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.namedWindow("License Plate Detector", cv2.WINDOW_NORMAL)
        cv2.imshow("License Plate Detector", frame)
        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("testing1.mp4")


