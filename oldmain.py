# import cv2
# import torch
# import numpy as np
# import time
# from concurrent.futures import ThreadPoolExecutor
# import collections
# from ultralytics import YOLO
# import sys
# from ocr_utils import get_stable_text, ocr_thread_function
# from process_enhanced_plate import submit_plate_for_enhancement
# sys.path.append("./sort")
# from sort import Sort  # Import SORT tracker
#
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# # Load YOLO model
# model = YOLO("license_plate_detector.pt").to(device)
#
# # SORT Tracker with optimized settings
# tracker = Sort(max_age=10, min_hits=5, iou_threshold=0.3)
#
# video_path = "testing.mp4"
# cap = cv2.VideoCapture(video_path)
#
# frame_count = 0  # Frame counter
# fps_queue = collections.deque(maxlen=10)  # Moving average FPS
#
# executor = ThreadPoolExecutor(max_workers=2)
#
# # enhanced_plates = {}
#
# while cap.isOpened():
#     start_time = time.time()  # Start FPS timer
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     with torch.no_grad():  # Disable gradients for faster YOLO inference
#         results = model(frame,imgsz=1280,device=device)
#
#     detections = []
#     for result in results:
#         for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
#             x1, y1, x2, y2 = map(int, box)
#             confidence = conf.item()
#             if confidence > 0.5:
#                 detections.append([x1, y1, x2, y2, confidence])
#
#     detections = np.array(detections) if detections else np.empty((0, 5))
#     tracked_objects = tracker.update(detections)
#
#     for obj in tracked_objects:
#         x1, y1, x2, y2, track_id = map(int, obj)
#
#         plate = frame[y1:y2, x1:x2]
#         if plate.size == 0:
#             continue
#
#         cv2.imshow(f"Cropped Plate {track_id}", plate)
#         #
#         # if frame_count % 60 == 0:
#         #     future = submit_plate_for_enhancement(plate)
#         #     enhanced_plates[track_id] = future 
#         #
#         # if track_id in enhanced_plates:
#         #     future = enhanced_plates[track_id]
#         #     if future.done():  # Check if the enhancement is finished
#         #         enhanced_plate = future.result()
#         #         cv2.imshow(f"Enhanced Plate {track_id}", enhanced_plate)
#
#         executor.submit(ocr_thread_function, plate, track_id)
#
#         final_plate_text = get_stable_text(track_id)
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID {track_id}: {final_plate_text}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0,255), 2)
#
#     frame_count += 1  # Increment frame count
#
#     # Calculate moving average FPS
#     fps = 1.0 / (time.time() - start_time)
#     fps_queue.append(fps)
#     avg_fps = sum(fps_queue) / len(fps_queue)
#
#     # Display FPS on frame
#     cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#
#     # Show frame
#     cv2.namedWindow("License Plate Detector", cv2.WINDOW_NORMAL)
#     cv2.imshow("License Plate Detector", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


