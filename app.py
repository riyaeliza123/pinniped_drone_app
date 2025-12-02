import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import cv2
import tempfile
import gc
import supervision as sv
import inspect
import psutil
from collections import defaultdict
from math import cos, radians
from pathlib import Path

from scripts.detection import run_detection_from_local
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary

st.title("Pinniped Detection from Drone Imagery")

st.markdown("### ⚙️ Detection Thresholds")
conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap / NMS threshold (%)", 0, 100, 30, step=5)

st.markdown("### 📁 Select Images")
folder_path = st.text_input("Enter folder path with drone images:", value="")

if folder_path and os.path.isdir(folder_path):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(folder_path).glob(ext))
    
    # Remove duplicates (case-insensitive file systems return same file twice)
    image_files = list(set(str(f) for f in image_files))
    
    if not image_files:
        st.warning("No images found in folder.")
        st.stop()
    
    st.success(f"Found {len(image_files)} images")
    
    if st.button("Process Images"):
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        peak_memory = initial_memory
        
        if 'out_dir' not in st.session_state:
            st.session_state['out_dir'] = tempfile.mkdtemp()
        out_dir = st.session_state['out_dir']

        st.session_state['annotated_paths'] = []
        st.session_state['all_detections_records'] = []
        st.session_state['summary_records'] = []
        st.session_state['grouped_coords'] = defaultdict(list)
        st.session_state['max_counts'] = defaultdict(int)

        total_files = len(image_files)
        progress_bar = st.progress(0)
        progress_label = st.empty()
        status_text = st.empty()

        for idx, image_path in enumerate(image_files):
            p1 = None
            p2 = None
            
            try:
                filename = os.path.basename(image_path)
                status_text.markdown(f"**Processing {idx + 1}/{total_files}**: {filename}")

                # Extract EXIF
                lat, lon = extract_gps_from_image(image_path)
                date, time = get_capture_date_time(image_path)

                if lat is None or lon is None:
                    st.warning(f"No GPS in {filename}. Skipped.")
                    continue

                # Resize for Roboflow limits
                p1, s1 = limit_resolution_to_temp(image_path)
                p2, s2 = progressive_resize_to_temp(p1)
                final_path = p2 or p1 or image_path
                scale = (s1 * s2) if p2 else (s1 if p1 else 1.0)

                # Run detection from local file
                detections = run_detection_from_local(final_path, conf_threshold, overlap_threshold)

                # Read for annotation BEFORE cleanup
                img = cv2.imread(final_path)
                if img is None:
                    raise RuntimeError(f"Could not read {filename}")

                # NOW cleanup resize temps (after reading)
                for p in (p2, p1):
                    if p and p != image_path and os.path.exists(p):
                        try:
                            os.remove(p)
                        except:
                            pass

                gsd = compute_gsd({}, img.shape[1])

                annotator = sv.BoxAnnotator()
                texts = [f"seal {c*100:.1f}%" for c in detections.confidence]
                labeled = annotator.annotate(scene=img.copy(), detections=detections, labels=texts) \
                    if "labels" in inspect.signature(annotator.annotate).parameters \
                    else annotator.annotate(scene=img.copy(), detections=detections)

                ann_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                ann_path = ann_tmp.name
                ann_tmp.close()
                cv2.imwrite(ann_path, labeled)
                st.session_state['annotated_paths'].append(ann_path)

                # Scale boxes back to original coordinates
                scaled_boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale] for b in detections.xyxy]

                for seal_idx, (box, conf) in enumerate(zip(scaled_boxes, detections.confidence), start=1):
                    st.session_state['all_detections_records'].append({
                        "image_name": filename,
                        "seal_id": seal_idx,
                        "x_min": float(box[0]),
                        "y_min": float(box[1]),
                        "x_max": float(box[2]),
                        "y_max": float(box[3]),
                        "confidence": float(conf)
                    })

                location = get_location_name(lat)
                group_key = (location, date)

                for box in detections.xyxy:
                    xc = (box[0] + box[2]) / 2
                    yc = (box[1] + box[3]) / 2
                    dx_m = (xc - img.shape[1] / 2) * gsd
                    dy_m = (yc - img.shape[0] / 2) * gsd
                    lat_off = dy_m / 111320
                    lon_off = dx_m / (111320 * cos(radians(lat)))
                    st.session_state['grouped_coords'][group_key].append((lat + lat_off, lon + lon_off))

                st.session_state['summary_records'].append({
                    "image_name": filename,
                    "pinniped_count": len(scaled_boxes),
                    "latitude": lat,
                    "longitude": lon,
                    "location": location,
                    "date": date,
                    "time": time
                })

                key = (location, date)
                current_max = st.session_state['max_counts'].get(key, 0)
                st.session_state['max_counts'][key] = max(current_max, len(scaled_boxes))

                del img, labeled
                gc.collect()

                current_memory = process.memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)

                progress_bar.progress((idx + 1) / total_files)
                progress_label.text(f"{idx + 1}/{total_files} complete")

            except Exception as e:
                st.error(f"Error processing {filename}: {e}")
                import traceback
                st.text(traceback.format_exc())
                
                # Cleanup on error
                for p in (p2, p1):
                    if p and p != image_path and os.path.exists(p):
                        try:
                            os.remove(p)
                        except:
                            pass
                gc.collect()

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_used = final_memory - initial_memory

        status_text.markdown(f"**✅ Complete! {len(st.session_state['summary_records'])}/{total_files} processed**")

        st.markdown("### 📊 Memory Usage")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak RAM", f"{peak_memory:.1f} MB")
        with col2:
            st.metric("Final RAM", f"{final_memory:.1f} MB")
        with col3:
            st.metric("RAM Increase", f"{memory_used:.1f} MB")

        st.markdown("### Annotated Images")
        for idx, ap in enumerate(st.session_state.get('annotated_paths', [])):
            try:
                # Get corresponding filename from summary records
                if idx < len(st.session_state['summary_records']):
                    filename = st.session_state['summary_records'][idx]['image_name']
                else:
                    filename = f"Image {idx + 1}"
                
                st.image(cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB))
                st.markdown(f"<p style='text-align: center; color: gray; font-size: 12px; margin-top: -15px;'>Image: {filename}</p>", unsafe_allow_html=True)
            except:
                st.image(ap)
                if idx < len(st.session_state['summary_records']):
                    filename = st.session_state['summary_records'][idx]['image_name']
                    st.markdown(f"<p style='text-align: center; color: gray; font-size: 12px; margin-top: -15px;'>Image: {filename}</p>", unsafe_allow_html=True)

        folder_summary_records = compute_unique_counts(
            st.session_state['grouped_coords'],
            st.session_state['max_counts']
        )
        display_and_download_summary(
            st.session_state['summary_records'],
            folder_summary_records,
            st.session_state['all_detections_records']
        )
elif folder_path:
    st.error("Invalid folder path")