import streamlit as st
import tempfile, os, cv2, gc
from collections import defaultdict
from math import cos, radians

from scripts.detection import run_detection, run_batch_detection
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
    import shutil

    out_dir = tempfile.mkdtemp()
    grouped_coords, max_counts, all_groups = defaultdict(list), defaultdict(int), set()
    all_detections_records, summary_records = [], []

    # ----------------------
    # Streaming preprocessing (process smaller batches to reduce memory)
    # ----------------------
    PREPROCESS_BATCH_SIZE = 5  # Preprocess 5 at a time
    DETECTION_BATCH_SIZE = 3   # Detect 3 at a time (smaller for large 40MB images)
    
    # First pass: collect metadata without loading all images
    st.info(f"📥 Preprocessing {len(uploaded_files)} images... This may take a few minutes.")
    progress_preprocess = st.progress(0)
    
    all_preprocessed = []
    
    def preprocess_image(uploaded):
        tmp_path = os.path.join(out_dir, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        p1, s1 = limit_resolution_to_temp(tmp_path)
        p2, s2 = progressive_resize_to_temp(p1)
        scale = s1 * s2
        lat, lon = extract_gps_from_image(tmp_path)
        date, time = get_capture_date_time(tmp_path)
        location = get_location_name(lat)
        
        # Only load full image for display later, save metadata now
        return {
            "name": uploaded.name,
            "tmp_path": tmp_path,
            "resized_path": p2,
            "scale": scale,
            "lat": lat,
            "lon": lon,
            "date": date,
            "time": time,
            "location": location,
            "group_key": (location, date)
        }

    # Preprocess in small batches to control memory
    for preprocess_batch_start in range(0, len(uploaded_files), PREPROCESS_BATCH_SIZE):
        preprocess_batch_end = min(preprocess_batch_start + PREPROCESS_BATCH_SIZE, len(uploaded_files))
        preprocess_batch = uploaded_files[preprocess_batch_start:preprocess_batch_end]
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            batch_results = list(executor.map(preprocess_image, preprocess_batch))
        
        all_preprocessed.extend(batch_results)
        progress_preprocess.progress(preprocess_batch_end / len(uploaded_files))
    
    progress_preprocess.empty()

    # ----------------------
    # Batch Detection loop with memory management
    # ----------------------
    st.info(f"🔍 Running detection on {len(all_preprocessed)} images in batches...")
    progress_detection = st.progress(0)
    
    for batch_start in range(0, len(all_preprocessed), DETECTION_BATCH_SIZE):
        batch_end = min(batch_start + DETECTION_BATCH_SIZE, len(all_preprocessed))
        batch_data = all_preprocessed[batch_start:batch_end]
        
        # Filter images with valid GPS data
        valid_batch = [data for data in batch_data if data["lat"] is not None and data["lon"] is not None]
        invalid_batch = [data for data in batch_data if data["lat"] is None or data["lon"] is None]
        
        # Show warnings for invalid images
        for data in invalid_batch:
            st.warning(f"⚠️ No GPS data in {data['name']}. Skipping.")
        
        if not valid_batch:
            progress_detection.progress(batch_end / len(all_preprocessed))
            continue
        
        # Run batch detection
        image_paths = [data["resized_path"] for data in valid_batch]
        detections_batch = run_batch_detection(image_paths, conf_threshold, overlap_threshold)
        
        # Process results from batch
        for data, detections in zip(valid_batch, detections_batch):
            # Load image only when needed (for annotation)
            img = cv2.imread(data["resized_path"])
            if img is None:
                st.warning(f"Could not read image {data['name']}")
                continue
            
            gsd = compute_gsd({}, img.shape[1])
            detections.xyxy *= data["scale"]
            
            count = len(detections.xyxy)
            max_counts[data["group_key"]] = max(max_counts[data["group_key"]], count)

            # Populate grouped_coords for clustering
            for box in detections.xyxy:
                xc = (box[0] + box[2]) / 2
                yc = (box[1] + box[3]) / 2
                dx_m = (xc - img.shape[1] / 2) * gsd
                dy_m = (yc - img.shape[0] / 2) * gsd
                lat_off = dy_m / 111320
                lon_off = dx_m / (111320 * cos(radians(data["lat"])))
                grouped_coords[data["group_key"]].append(((data["lat"] + lat_off, data["lon"] + lon_off), gsd, data["time"]))

            # Annotate and display
            annotator = sv.BoxAnnotator()
            texts = [f"seal {conf*100:.1f}%" for conf in detections.confidence]
            labeled = annotator.annotate(scene=img.copy(), detections=detections, labels=texts) \
                if "labels" in inspect.signature(annotator.annotate).parameters \
                else annotator.annotate(scene=img.copy(), detections=detections)
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
            
            # Free memory after processing
            del img
        
        progress_detection.progress(batch_end / len(all_preprocessed))
        gc.collect()  # Force garbage collection after each batch
    
    progress_detection.empty()

    # Cleanup temp files
    shutil.rmtree(out_dir, ignore_errors=True)

    folder_summary_records = compute_unique_counts(grouped_coords, max_counts)
    display_and_download_summary(summary_records, folder_summary_records, all_detections_records)
