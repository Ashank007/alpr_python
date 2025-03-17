import time
import collections

fps_queue = collections.deque(maxlen=10)

def calculate_fps(start_time):
    """Calculates FPS using a moving average."""
    fps = 1.0 / (time.time() - start_time)
    fps_queue.append(fps)
    return sum(fps_queue) / len(fps_queue)

