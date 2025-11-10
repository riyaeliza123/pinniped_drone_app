import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import exifread
from datetime import datetime
from math import cos, radians
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyproj import Transformer
from roboflow import Roboflow
import streamlit as st
import supervision as sv
import inspect

# -------------------------------
# CONFIGURATION
# -------------------------------
API_KEY = st.secrets["ROBOWFLOW_API_KEY"]
PROJECT_NAME = "pinnipeds-drone-imagery"
MODEL_VERSION = 18
MAX_PIXELS = 4_000_000
MAX_SIZE_MB = 15
MIN_SCALE_PERCENT = 10

CAMERA_SENSOR_WIDTHS = {
    "L2D-20c": 13.2,
    "FC3411": 13.2,
    "FC220": 6.3,
}

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_NAME)
model = project.version(MODEL_VERSION).model

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_float(val):
    try:
        if hasattr(val, "values"):
            val = val.values[0]
        if hasattr(val, "num") and hasattr(val, "den"):
            return float(val.num) / float(val.den)
        if isinstance(val, str) and "/" in val:
            num, den = map(float, val.split("/"))
            return num / den
        return float(val)
    except:
        return None

def dms_to_decimal(dms, ref):
    degrees = get_float(dms[0])
    minutes = get_float(dms[1])
    seconds = get_float(dms[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal

def extract_gps_from_image(path):
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
        lat = dms_to_decimal(tags["GPS GPSLatitude"].values, tags["GPS GPSLatitudeRef"].values)
        lon = dms_to_decimal(tags["GPS GPSLongitude"].values, tags["GPS GPSLongitudeRef"].values)
        return lat, lon
    return None, None

def get_location_name(lat):
    if 48 <= lat < 49:
        return "Cowichan"
    elif 49 <= lat < 50:
        return "Nanaimo"
    elif 50 <= lat < 51:
        return "Campbell River"
    else:
        return "Unknown"

def get_capture_date_time(path):
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if "Image DateTime" in tags:
        raw = str(tags["Image DateTime"])
        dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        return dt.date(), dt.time()
    return None, None

def compute_gsd(tags, img_width):
    alt = get_float(tags.get("GPS GPSAltitude"))
    focal = get_float(tags.get("EXIF FocalLength"))
    if alt is None: alt = 20.0
    if focal is None: focal = 24.0
    model_name = str(tags.get("Image Model") or "Unknown").strip()
    sensor_w = CAMERA_SENSOR_WIDTHS.get(model_name, 13.2)
    if focal == 0 or img_width == 0:
        return float("inf")
    return (alt * sensor_w) / (focal * img_width)

def limit_resolution_to_temp(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    pixel_count = h * w

    if pixel_count <= MAX_PIXELS:
        return image_path, 1.0

    scale_percent = 80
    while scale_percent >= MIN_SCALE_PERCENT:
        width = int(w * scale_percent / 100)
        height = int(h * scale_percent / 100)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)  # close file descriptor immediately
        cv2.imwrite(temp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if resized.shape[0] * resized.shape[1] <= MAX_PIXELS:
            return temp_path, (w / width)

        os.remove(temp_path)
        scale_percent -= 10

    return temp_path, (w / width)

def progressive_resize_to_temp(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale_percent = 80
    while scale_percent >= MIN_SCALE_PERCENT:
        new_w = int(w * scale_percent / 100)
        new_h = int(h * scale_percent / 100)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        if size_mb <= MAX_SIZE_MB:
            return tmp.name, (w / new_w)
        os.remove(tmp.name)
        scale_percent -= 10
    return tmp.name, (w / new_w)

def parse_roboflow_detections(result_json):
    xyxy, conf, cid = [], [], []
    for pred in result_json.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        conf.append(pred["confidence"])
        cid.append(0)
    if not xyxy:
        return sv.Detections(
            xyxy=np.zeros((0, 4)),
            confidence=np.array([]),
            class_id=np.array([])
        )
    return sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(conf),
        class_id=np.array(cid)
    )

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Drone Imagery Pinniped Detection")
st.markdown("Upload drone images and detect pinnipeds using a YOLOv11 model (Roboflow).")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.markdown("### ⚙️ Detection Thresholds")

conf_threshold = st.slider(
    "Minimum confidence threshold (%)",
    min_value=0, max_value=100, value=15, step=5,
    help="Lower = more detections (higher recall), higher = fewer false positives."
)

overlap_threshold = st.slider(
    "Overlap (NMS) threshold (%)",
    min_value=0, max_value=100, value=30, step=5,
    help="Higher = keeps overlapping detections separate; lower = merges close boxes."
)

if uploaded_files:
    progress = st.progress(0)
    summary_records = []
    grouped_coords = defaultdict(list)
    max_counts = defaultdict(int)
    all_groups = set()
    out_dir = tempfile.mkdtemp()

    all_detections_records = []

    for i, uploaded in enumerate(uploaded_files):
        # Save to temp file
        tmp_path = os.path.join(out_dir, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        img = cv2.imread(tmp_path)
        with open(tmp_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        lat, lon = extract_gps_from_image(tmp_path)
        if lat is None or lon is None:
            st.warning(f"No GPS data found in {uploaded.name}. Skipping.")
            continue

        location = get_location_name(lat)
        date, time = get_capture_date_time(tmp_path)
        gsd = compute_gsd(tags, img.shape[1])
        group_key = (location, date)
        all_groups.add(group_key)

        # Resize
        p1, s1 = limit_resolution_to_temp(tmp_path)
        p2, s2 = progressive_resize_to_temp(p1)
        scale = s1 * s2

        # Run inference with user-defined thresholds
        result = model.predict(
            p2,
            confidence=conf_threshold,
            overlap=overlap_threshold
        ).json()


        detections = parse_roboflow_detections(result)
        detections.xyxy *= scale
        roboflow_count = len(detections.xyxy)
        max_counts[group_key] = max(max_counts[group_key], roboflow_count)

        annotator = sv.BoxAnnotator()

        texts = [f"seal {conf*100:.1f}%" for conf in detections.confidence] if len(detections) > 0 else []

        if "labels" in inspect.signature(annotator.annotate).parameters:
            labeled = annotator.annotate(
                scene=img.copy(),
                detections=detections,
                labels=texts
            )
        else:
            labeled = annotator.annotate(
                scene=img.copy(),
                detections=detections,
                text=texts
            )

        save_path = os.path.join(out_dir, f"annotated_{uploaded.name}")
        cv2.imwrite(save_path, labeled)

        # Display annotated
        st.image(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB), caption=f"Annotated: {uploaded.name}")

        # Create per-image detection details
        if len(detections.xyxy) > 0:
            for idx, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence), start=1):
                all_detections_records.append({
                    "image_name": uploaded.name,
                    "seal_id": idx,
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3],
                    "confidence": conf
                })

        # Record summary
        summary_records.append({
            "image_name": uploaded.name,
            "pinniped_count": roboflow_count,
            "latitude": lat,
            "longitude": lon,
            "location": location,
            "date": date,
            "time": time
        })

        # Collect detections for clustering
        for box in detections.xyxy:
            xc = (box[0] + box[2]) / 2
            yc = (box[1] + box[3]) / 2
            dx_m = (xc - img.shape[1] / 2) * gsd
            dy_m = (yc - img.shape[0] / 2) * gsd
            lat_off = dy_m / 111320
            lon_off = dx_m / (111320 * cos(radians(lat)))
            grouped_coords[group_key].append(((lat + lat_off, lon + lon_off), gsd, time))

        progress.progress((i + 1) / len(uploaded_files))

    if all_detections_records:
        all_detections_df = pd.DataFrame(all_detections_records)
        st.download_button(
            label="⬇️ Download Confidence of Pinniped detections CSV",
            data=all_detections_df.to_csv(index=False),
            file_name="all_pinniped_detection_conf.csv",
            mime="text/csv"
        )
    else:
        st.info("No detections found in any uploaded images.")

    # Deduplicate
    st.subheader("📊 Unique Pinniped Estimates")
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    folder_summary_records = []
    for (loc, date) in all_groups:
        coords_data = grouped_coords.get((loc, date), [])
        if not coords_data:
            folder_summary_records.append({"survey_location": loc, "date": date, "unique_pinniped_count": 0})
            continue
        coords, gsd_values, times = zip(*coords_data)
        coords_xy = np.array([transformer.transform(lon, lat) for lat, lon in coords])
        if len(coords_xy) == 1:
            count = 1
        else:
            nn = NearestNeighbors(n_neighbors=2).fit(coords_xy)
            distances, _ = nn.kneighbors(coords_xy)
            median_dist = np.median(distances[:, 1])
            eps = np.clip(median_dist * 1.75, 1.7, 3.65)
            clustering = DBSCAN(eps=eps, min_samples=1).fit(coords_xy)
            count = len(set(clustering.labels_))
        count = max(count, max_counts.get((loc, date), 0))
        folder_summary_records.append({
            "survey_location": loc,
            "date": date,
            "unique_pinniped_count": count
        })
        st.write(f"**{loc} — {date}:** {count} unique pinnipeds")

    summary_df = pd.DataFrame(summary_records)
    folder_df = pd.DataFrame(folder_summary_records)
    st.subheader("Per-Image Summary")
    st.dataframe(summary_df)
    st.download_button("Download Image Summary CSV", summary_df.to_csv(index=False), "image_summary.csv")

    st.subheader("Per-Location Summary")
    st.dataframe(folder_df)
    st.download_button("Download Location Summary CSV", folder_df.to_csv(index=False), "unique_counts.csv")


