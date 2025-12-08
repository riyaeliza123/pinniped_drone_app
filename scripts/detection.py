# Run Roboflow model inference and convert JSON to Supervision format
import os
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient

def _get_api_key():
    """Get Roboflow API key from secrets or environment"""
    api_key = None
    
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("ROBOFLOW_API_KEY") or st.secrets.get("ROBOWFLOW_API_KEY")
    except Exception:
        pass
    
    if not api_key:
        api_key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("ROBOWFLOW_API_KEY")
    
    if not api_key:
        raise RuntimeError("Roboflow API key not found in secrets or environment")
    
    return api_key

API_KEY = _get_api_key()
MODEL_ID = "pinnipeds-drone-imagery/18"

_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / max(a_area + b_area - inter, 1e-9)

def _nms(boxes, scores, iou_thresh):
    if not boxes:
        return [], []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(boxes[i], boxes[j]) <= iou_thresh]
    return [boxes[k] for k in keep], [scores[k] for k in keep]

def parse_roboflow_detections(result_json):
    xyxy, conf, cid = [], [], []
    for pred in result_json.get("predictions", []):
        try:
            x, y, w, h = float(pred["x"]), float(pred["y"]), float(pred["width"]), float(pred["height"])
            c = float(pred.get("confidence", 0.0))
        except Exception:
            continue
        xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        conf.append(c)
        cid.append(0)
    if not xyxy:
        return sv.Detections(xyxy=np.zeros((0, 4)), confidence=np.array([]), class_id=np.array([]))
    return sv.Detections(xyxy=np.array(xyxy), confidence=np.array(conf), class_id=np.array(cid))

def run_detection_from_local(image_path, confidence_percent, overlap_percent):
    """Run detection from local file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Ensure JPEG
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")
    
    base, ext = os.path.splitext(image_path)
    if ext.lower() not in ['.jpg', '.jpeg']:
        jpg_path = base + ".jpg"
        cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        inference_path = jpg_path
    else:
        inference_path = image_path
    
    try:
        result = _client.infer(inference_path, model_id=MODEL_ID)
    except Exception as e:
        raise RuntimeError(f"Roboflow inference failed: {e}")

    det = parse_roboflow_detections(result)
    conf_cut = confidence_percent / 100.0
    iou_cut = overlap_percent / 100.0

    mask = det.confidence >= conf_cut
    det = sv.Detections(
        xyxy=det.xyxy[mask],
        confidence=det.confidence[mask],
        class_id=(det.class_id[mask] if det.class_id.size else np.zeros(np.sum(mask), dtype=int))
    )

    boxes = det.xyxy.tolist()
    scores = det.confidence.tolist()
    boxes, scores = _nms(boxes, scores, iou_cut)
    
    if not boxes:
        return sv.Detections(xyxy=np.zeros((0, 4)), confidence=np.array([]), class_id=np.array([]))
    return sv.Detections(xyxy=np.array(boxes), confidence=np.array(scores), class_id=np.zeros(len(boxes), dtype=int))
