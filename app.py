import streamlit as st
import tempfile, os, cv2
from collections import defaultdict
from math import cos, radians

from scripts.detection import run_detection
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary
import supervision as sv
import inspect

st.write("DEPLOYED KEY:", "ROBOWFLOW_API_KEY" in st.secrets)

st.title("Pinniped Detection from Drone Imagery")
st.markdown("Upload drone images to detect seals using a YOLOv11 model (via Roboflow).")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.markdown("### ⚙️ Detection Thresholds")

conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap threshold (%)", 0, 100, 30, step=5)

if uploaded_files:
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    out_dir = tempfile.mkdtemp()
    grouped_coords, max_counts, all_groups = defaultdict(list), defaultdict(int), set()
    all_detections_records, summary_records = [], []

    # ----------------------
    # Parallel preprocessing
    # ----------------------
    def preprocess_image(uploaded):
        tmp_path = os.path.join(out_dir, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        p1, s1 = limit_resolution_to_temp(tmp_path)
        p2, s2 = progressive_resize_to_temp(p1)
        scale = s1 * s2
        img = cv2.imread(tmp_path)
        lat, lon = extract_gps_from_image(tmp_path)
        date, time = get_capture_date_time(tmp_path)
        location = get_location_name(lat)
        gsd = compute_gsd({}, img.shape[1])

        group_key = (location, date)
        all_groups.add(group_key)

        return {
            "name": uploaded.name,
            "tmp_path": tmp_path,
            "resized_path": p2,
            "scale": scale,
            "img": img,
            "lat": lat,
            "lon": lon,
            "date": date,
            "time": time,
            "location": location,
            "gsd": gsd,
            "group_key": group_key
        }

    # Run preprocessing concurrently
    with ThreadPoolExecutor() as executor:
        preprocessed_results = list(executor.map(preprocess_image, uploaded_files))

    # ----------------------
    # Detection loop
    # ----------------------
    progress = st.progress(0)
    for i, data in enumerate(preprocessed_results):
        if data["lat"] is None or data["lon"] is None:
            st.warning(f"No GPS data in {data['name']}. Skipping.")
            continue

        detections = run_detection(data["resized_path"], conf_threshold, overlap_threshold)
        detections.xyxy *= data["scale"]

        count = len(detections.xyxy)
        max_counts[data["group_key"]] = max(max_counts[data["group_key"]], count)

        # Populate grouped_coords for clustering
        for box in detections.xyxy:
            xc = (box[0] + box[2]) / 2
            yc = (box[1] + box[3]) / 2
            dx_m = (xc - data["img"].shape[1] / 2) * data["gsd"]
            dy_m = (yc - data["img"].shape[0] / 2) * data["gsd"]
            lat_off = dy_m / 111320
            lon_off = dx_m / (111320 * cos(radians(data["lat"])))
            grouped_coords[data["group_key"]].append(((data["lat"] + lat_off, data["lon"] + lon_off), data["gsd"], data["time"]))

        # Annotate and display
        annotator = sv.BoxAnnotator()
        texts = [f"seal {conf*100:.1f}%" for conf in detections.confidence]
        labeled = annotator.annotate(scene=data["img"].copy(), detections=detections, labels=texts) \
            if "labels" in inspect.signature(annotator.annotate).parameters \
            else annotator.annotate(scene=data["img"].copy(), detections=detections)
        st.image(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB), caption=f"Annotated: {data['name']}")

        # Save detections
        for idx, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence), start=1):
            all_detections_records.append({
                "image_name": data["name"],
                "seal_id": idx,
                "x_min": box[0],
                "y_min": box[1],
                "x_max": box[2],
                "y_max": box[3],
                "confidence": conf
            })

        summary_records.append({
            "image_name": data["name"],
            "pinniped_count": count,
            "latitude": data["lat"],
            "longitude": data["lon"],
            "location": data["location"],
            "date": data["date"],
            "time": data["time"]
        })

        progress.progress((i + 1) / len(preprocessed_results))

    folder_summary_records = compute_unique_counts(grouped_coords, max_counts)
    display_and_download_summary(summary_records, folder_summary_records, all_detections_records)
