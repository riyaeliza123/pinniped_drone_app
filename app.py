import streamlit as st
import tempfile, os, cv2, gc
from collections import defaultdict
from math import cos, radians

from scripts.detection import run_detection
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary
import supervision as sv
import inspect

st.title("Pinniped Detection from Drone Imagery")
st.markdown("Upload drone images to detect seals using a YOLOv11 model (via Roboflow).")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.markdown("### ⚙️ Detection Thresholds")

conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap threshold (%)", 0, 100, 30, step=5)

if uploaded_files:
    import shutil

    out_dir = tempfile.mkdtemp()
    grouped_coords, max_counts = defaultdict(list), defaultdict(int)
    all_detections_records, summary_records = [], []

    st.info(f"📥 Processing {len(uploaded_files)} images sequentially...")
    progress_text = st.empty()
    
    annotated_count = 0
    
    for idx, uploaded in enumerate(uploaded_files):
        # Update progress counter
        progress_text.markdown(f"**Progress: {annotated_count}/{len(uploaded_files)} annotated**")
        
        try:
            # Save uploaded file
            tmp_path = os.path.join(out_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            # Preprocess: resize
            p1, s1 = limit_resolution_to_temp(tmp_path)
            p2, s2 = progressive_resize_to_temp(p1)
            scale = s1 * s2

            # Extract metadata
            lat, lon = extract_gps_from_image(tmp_path)
            date, time = get_capture_date_time(tmp_path)
            location = get_location_name(lat)
            
            # Check GPS validity
            if lat is None or lon is None:
                st.warning(f"⚠️ No GPS data in {uploaded.name}. Skipping.")
                continue

            group_key = (location, date)

            # Run detection
            detections = run_detection(p2, conf_threshold, overlap_threshold)

            # Load image for annotation (resized image p2)
            img = cv2.imread(p2)
            if img is None:
                st.warning(f"❌ Could not read image {uploaded.name}")
                # make sure we still try to remove temp files for this image
                raise RuntimeError("Failed to read resized image")

            gsd = compute_gsd({}, img.shape[1])

            # Do NOT modify detections in-place before annotating resized image.
            # The model returned coordinates relative to the resized image `p2`.
            # Annotate using those coordinates. For saving/export, compute
            # scaled coordinates that map back to the original image size.

            count = len(detections.xyxy)
            max_counts[group_key] = max(max_counts[group_key], count)

            # Populate grouped_coords for clustering using resized-image coordinates
            for box in detections.xyxy:
                # box is [x_min, y_min, x_max, y_max] in resized image pixels
                xc = (box[0] + box[2]) / 2
                yc = (box[1] + box[3]) / 2
                dx_m = (xc - img.shape[1] / 2) * gsd
                dy_m = (yc - img.shape[0] / 2) * gsd
                lat_off = dy_m / 111320
                lon_off = dx_m / (111320 * cos(radians(lat)))
                grouped_coords[group_key].append(((lat + lat_off, lon + lon_off), gsd, time))

            # Annotate and display
            annotator = sv.BoxAnnotator()
            texts = [f"seal {conf*100:.1f}%" for conf in detections.confidence]
            labeled = annotator.annotate(scene=img.copy(), detections=detections, labels=texts) \
                if "labels" in inspect.signature(annotator.annotate).parameters \
                else annotator.annotate(scene=img.copy(), detections=detections)
            st.image(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB), caption=f"✅ {uploaded.name} ({count} seals)")

            # Save detections: scale coordinates back to original image size
            scaled_boxes = (detections.xyxy * scale) if hasattr(detections.xyxy, "__mul__") else detections.xyxy
            for seal_idx, (box, conf) in enumerate(zip(scaled_boxes, detections.confidence), start=1):
                all_detections_records.append({
                    "image_name": uploaded.name,
                    "seal_id": seal_idx,
                    "x_min": float(box[0]),
                    "y_min": float(box[1]),
                    "x_max": float(box[2]),
                    "y_max": float(box[3]),
                    "confidence": float(conf)
                })

            summary_records.append({
                "image_name": uploaded.name,
                "pinniped_count": count,
                "latitude": lat,
                "longitude": lon,
                "location": location,
                "date": date,
                "time": time
            })

            annotated_count += 1
            progress_text.markdown(f"**Progress: {annotated_count}/{len(uploaded_files)} annotated**")
            
            # Free memory
            del img
            gc.collect()

        except Exception as e:
            st.error(f"❌ Error processing {uploaded.name}: {str(e)}")
            # continue to next image after ensuring temp cleanup below
        finally:
            # Per-image cleanup: remove original upload and intermediate resized files
            try:
                if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            try:
                if 'p1' in locals() and p1 and os.path.exists(p1) and p1 != tmp_path:
                    os.remove(p1)
            except Exception:
                pass
            try:
                if 'p2' in locals() and p2 and os.path.exists(p2) and p2 not in (tmp_path, p1):
                    os.remove(p2)
            except Exception:
                pass
            gc.collect()
            # ensure loop continues
            continue

    # Cleanup temp files
    shutil.rmtree(out_dir, ignore_errors=True)

    # Final summary
    progress_text.markdown(f"**✅ Complete! {annotated_count}/{len(uploaded_files)} annotated**")

    folder_summary_records = compute_unique_counts(grouped_coords, max_counts)
    display_and_download_summary(summary_records, folder_summary_records, all_detections_records)
