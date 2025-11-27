# Run Roboflow model inference and convert JSON to Supervision format
import os
import time
import random
import threading
import numpy as np
import supervision as sv
from scripts.config import model
import cv2
import hashlib


def demo_detection(image_path, confidence, overlap):
    """Return a deterministic set of demo detections for `image_path`.
    Produces a few boxes positioned across the image so the pipeline can be tested
    without calling Roboflow. The boxes are deterministic (based on a hash)
    so repeated runs look similar for the same image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return sv.Detections(xyxy=np.zeros((0, 4)), confidence=np.array([]), class_id=np.array([]))

    h, w = img.shape[:2]
    # derive a small integer seed from the path so results are deterministic
    seed = int(hashlib.md5(image_path.encode('utf-8')).hexdigest()[:8], 16)

    # choose 1-4 boxes based on seed
    n = 1 + (seed % 4)
    xyxy = []
    conf = []
    cid = []
    for i in range(n):
        # position boxes in different quadrants
        cx = int(((i + 1) * w) / (n + 1))
        cy = int(((i + 1) * h) / (n + 1))
        box_w = int(w * 0.15)
        box_h = int(h * 0.12)
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w - 1, cx + box_w // 2)
        y2 = min(h - 1, cy + box_h // 2)
        xyxy.append([x1, y1, x2, y2])
        # confidence varies but honors threshold-ish behavior
        conf_val = 0.6 + ((seed >> (i * 3)) % 40) / 100.0
        conf.append(conf_val)
        cid.append(0)

    return sv.Detections(xyxy=np.array(xyxy), confidence=np.array(conf), class_id=np.array(cid))

def parse_roboflow_detections(result_json):
    xyxy, conf, cid = [], [], []
    for pred in result_json.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        conf.append(pred["confidence"])
        cid.append(0)
    if not xyxy:
        return sv.Detections(xyxy=np.zeros((0, 4)), confidence=np.array([]), class_id=np.array([]))
    return sv.Detections(xyxy=np.array(xyxy), confidence=np.array(conf), class_id=np.array(cid))

def run_detection(image_path, confidence, overlap):
    # default direct Roboflow call (wrapped by _roboflow_predict_with_retries below)
    result = _roboflow_predict_with_retries(image_path, confidence, overlap)
    return parse_roboflow_detections(result)

def run_batch_detection(image_paths, confidence, overlap):
    """Process a batch of images and return detection results (rate-limited).

    This uses the same rate-limit and retry logic as `run_detection`.
    """
    results = []
    for path in image_paths:
        det = run_detection(path, confidence, overlap)
        results.append(det)
    return results


# --- Rate limiting and retry/backoff logic ---
# Config via environment variables with sensible defaults
RATE_LIMIT_PER_MINUTE = int(os.getenv("ROBOFLOW_RATE_LIMIT_PER_MINUTE", "60"))
MAX_RETRIES = int(os.getenv("ROBOFLOW_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("ROBOFLOW_BACKOFF_BASE", "1.0"))

# Internal rate-limiting state
_last_call_time = 0.0
_rate_lock = threading.Lock()


def _wait_for_rate_limit():
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    min_interval = 60.0 / float(RATE_LIMIT_PER_MINUTE)
    global _last_call_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_call_time = time.time()


def _roboflow_predict_with_retries(image_path, confidence, overlap):
    attempt = 0
    while True:
        try:
            _wait_for_rate_limit()
            result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
            return result
        except Exception:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            backoff = BACKOFF_BASE * (2 ** (attempt - 1))
            jitter = random.uniform(0, backoff * 0.1)
            time.sleep(backoff + jitter)
