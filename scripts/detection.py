# Run Roboflow model inference and convert JSON to Supervision format
import numpy as np
import supervision as sv
from scripts.config import model

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
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    return parse_roboflow_detections(result)

def run_batch_detection(image_paths, confidence, overlap):
    """Process a batch of images and return detection results."""
    results = []
    for path in image_paths:
        result = model.predict(path, confidence=confidence, overlap=overlap).json()
        results.append(parse_roboflow_detections(result))
    return results
