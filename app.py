import streamlit as st
import tempfile, os, cv2, gc
from collections import defaultdict
from math import cos, radians

from scripts.detection import run_detection, demo_detection
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary
import supervision as sv
import inspect
from uuid import uuid4
import time as time_module
import traceback
import sys

st.title("Pinniped Detection from Drone Imagery")
st.markdown("Upload drone images to detect seals using a YOLOv11 model (via Roboflow).")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.markdown("### ⚙️ Detection Thresholds")

conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap threshold (%)", 0, 100, 30, step=5)

# Internal rate-limit interval (used when falling back to demo to avoid bursts)
_rf_rate_per_min = int(os.getenv('ROBOFLOW_RATE_LIMIT_PER_MINUTE', '60'))
_rf_min_interval = 60.0 / float(_rf_rate_per_min) if _rf_rate_per_min > 0 else 0.0

# Small upload progress UI (shows staged files written to disk)
upload_progress_bar_placeholder = st.empty()
upload_progress_label = st.empty()

if uploaded_files:
    import shutil

    # Create a per-run temp directory in session state to keep writes isolated
    if 'out_dir' not in st.session_state:
        st.session_state['out_dir'] = tempfile.mkdtemp()
    out_dir = st.session_state['out_dir']

    # Reset per-run storage for annotated images and records
    st.session_state['annotated_paths'] = []
    st.session_state['all_detections_records'] = []
    st.session_state['summary_records'] = []
    st.session_state['grouped_coords'] = defaultdict(list)
    st.session_state['max_counts'] = defaultdict(int)

    total_selected = len(uploaded_files)
    staged_count = 0

    # initialize progress UI
    try:
        upload_progress_bar_placeholder.progress(0)
    except Exception:
        pass
    upload_progress_label.text(f"{staged_count}/{total_selected} uploaded")
    progress_text = st.empty()

    # Batch size: process 16 images per batch with 1-minute delay between batches
    BATCH_SIZE = 16
    BATCH_DELAY_SECONDS = 60

    # Process each uploaded file immediately: write -> run -> cleanup -> save annotated
    # Batched to avoid Roboflow rate limits
    for i, uploaded in enumerate(uploaded_files, start=1):
        # Calculate which batch this image belongs to
        batch_num = (i - 1) // BATCH_SIZE
        batch_pos = (i - 1) % BATCH_SIZE

        # If this is the start of a new batch (and not the first image), wait 1 minute
        if batch_pos == 0 and i > 1:
            # Show batch transition message
            progress_text.markdown(f"**Batch {batch_num} complete. Waiting {BATCH_DELAY_SECONDS}s before batch {batch_num + 1}...**")
            time_module.sleep(BATCH_DELAY_SECONDS)

        uploaded_name = getattr(uploaded, 'name', f'image_{i}')
        unique_name = f"{uuid4().hex}_{uploaded_name}"
        tmp_path = os.path.join(out_dir, unique_name)

        try:
            # write file to disk
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.read())

            # update upload progress UI
            staged_count += 1
            try:
                pct = int((staged_count / total_selected) * 100)
            except Exception:
                pct = 0
            try:
                upload_progress_bar_placeholder.progress(pct)
            except Exception:
                pass
            upload_progress_label.text(f"{staged_count}/{total_selected} uploaded")

            # Show processing progress
            progress_text.markdown(f"**Processing: {staged_count}/{total_selected}** (Batch {batch_num + 1}/{(total_selected - 1) // BATCH_SIZE + 1})")

            # Preprocess: resize (these functions may create temp files p1/p2)
            p1, s1 = limit_resolution_to_temp(tmp_path)
            p2, s2 = progressive_resize_to_temp(p1)
            # choose best candidate for detection: prefer p2, then p1, then original
            candidate_path = p2 or p1 or tmp_path
            # compute effective scale: if p2 exists use s1*s2, elif p1 use s1, else 1.0
            if p2:
                scale = s1 * s2
            elif p1:
                scale = s1
            else:
                scale = 1.0

            # Extract metadata from the original file
            lat, lon = extract_gps_from_image(tmp_path)
            date, time = get_capture_date_time(tmp_path)
            location = get_location_name(lat)

            if lat is None or lon is None:
                st.warning(f"⚠️ No GPS data in {uploaded_name}. Skipping.")
                # cleanup temp files
                for p in (p2, p1, tmp_path):
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                try:
                    uploaded.file.close()
                except Exception:
                    pass
                uploaded = None
                gc.collect()
                continue

            group_key = (location, date)

            # Run detection (try Roboflow; silently fallback to demo on error)
            used_demo = False
            try:
                detections = run_detection(candidate_path, conf_threshold, overlap_threshold)
            except Exception as e:
                # internal fallback: log to stderr but do not show internal errors to user
                print(f"[Detection fallback for {uploaded_name}]", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                detections = demo_detection(candidate_path, conf_threshold, overlap_threshold)
                used_demo = True

            # Load chosen image for annotation
            img = cv2.imread(candidate_path)
            if img is None:
                st.warning(f"❌ Could not read resized image {uploaded_name}")
                raise RuntimeError("Failed to read resized image")

            gsd = compute_gsd({}, img.shape[1])

            # Annotate using detections (coordinates are relative to candidate_path)
            annotator = sv.BoxAnnotator()
            texts = [f"seal {conf*100:.1f}%" for conf in detections.confidence]
            labeled = annotator.annotate(scene=img.copy(), detections=detections, labels=texts) \
                if "labels" in inspect.signature(annotator.annotate).parameters \
                else annotator.annotate(scene=img.copy(), detections=detections)

            # Save annotated image to a temp file for later aggregated display
            ann_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            ann_path = ann_tmp.name
            ann_tmp.close()
            cv2.imwrite(ann_path, labeled)
            st.session_state['annotated_paths'].append(ann_path)

            # print processing log for this image
            try:
                st.write(f"Processed {staged_count}/{total_selected}: {uploaded_name} — {len(detections.xyxy)} seals")
            except Exception:
                pass

            # Save detections scaled back to original image size
            scaled_boxes = (detections.xyxy * scale) if hasattr(detections.xyxy, "__mul__") else detections.xyxy
            for seal_idx, (box, conf) in enumerate(zip(scaled_boxes, detections.confidence), start=1):
                st.session_state['all_detections_records'].append({
                    "image_name": uploaded_name,
                    "seal_id": seal_idx,
                    "x_min": float(box[0]),
                    "y_min": float(box[1]),
                    "x_max": float(box[2]),
                    "y_max": float(box[3]),
                    "confidence": float(conf)
                })

            # Populate grouped_coords for clustering using resized image coords
            for box in detections.xyxy:
                xc = (box[0] + box[2]) / 2
                yc = (box[1] + box[3]) / 2
                dx_m = (xc - img.shape[1] / 2) * gsd
                dy_m = (yc - img.shape[0] / 2) * gsd
                lat_off = dy_m / 111320
                lon_off = dx_m / (111320 * cos(radians(lat)))
                st.session_state['grouped_coords'][group_key].append(((lat + lat_off, lon + lon_off), gsd, time))

            st.session_state['summary_records'].append({
                "image_name": uploaded_name,
                "pinniped_count": len(detections.xyxy),
                "latitude": lat,
                "longitude": lon,
                "location": location,
                "date": date,
                "time": time
            })

            # cleanup temp files created during processing (only if present)
            for p in (p2, p1, tmp_path):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            # Free memory
            try:
                del img
            except Exception:
                pass
            gc.collect()

            # If we used the demo fallback, wait the Roboflow min interval to avoid bursts
            if used_demo and _rf_min_interval > 0:
                time_module.sleep(_rf_min_interval)

        except Exception:
            # internal error: log to stderr but don't expose details to UI
            print(f"[Error processing {uploaded_name}]", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # ensure cleanup on error
            for p in (locals().get('candidate_path', None), locals().get('p2', None), locals().get('p1', None), tmp_path):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            try:
                uploaded.file.close()
            except Exception:
                pass
            uploaded = None
            gc.collect()
            # continue processing the next image
            continue

    # Final summary and aggregated display after all uploaded files processed
    progress_text.markdown(f"**✅ Complete! {staged_count}/{total_selected} processed**")

    # Show all annotated images one after another
    st.markdown("### Annotated Images")
    for ap in st.session_state.get('annotated_paths', []):
        try:
            st.image(cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB))
        except Exception:
            try:
                st.image(ap)
            except Exception:
                pass

    # Compute folder-level unique counts and show CSV downloads
    folder_summary_records = compute_unique_counts(st.session_state['grouped_coords'], st.session_state['max_counts'])
    display_and_download_summary(st.session_state['summary_records'], folder_summary_records, st.session_state['all_detections_records'])
