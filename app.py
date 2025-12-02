import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import cv2
import tempfile
import gc
import supervision as sv
import inspect
import psutil
import time
from collections import defaultdict
from math import cos, radians

from scripts.detection import run_detection_from_url, run_detection_from_local
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary
from scripts.s3_utils import (
    upload_to_s3_direct,
    generate_presigned_url,
    delete_from_s3,
    check_s3_object_exists,
    download_from_s3
)

try:
    S3_BUCKET = st.secrets["S3_BUCKET_NAME"]
except:
    S3_BUCKET = os.getenv("S3_BUCKET_NAME", "pinniped-drone-uploads")

st.title("Pinniped Detection from Drone Imagery")

st.markdown("### ⚙️ Detection Thresholds")
conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap / NMS threshold (%)", 0, 100, 30, step=5)

st.markdown("Upload drone images to detect seals using Roboflow serverless inference.")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

upload_progress_bar = st.empty()
upload_progress_label = st.empty()

if uploaded_files:
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

    total_files = len(uploaded_files)
    
    # Stage 1: Upload resized versions to S3
    st.info(f"Processing and uploading {total_files} files to S3...")
    s3_keys = []
    upload_progress_bar.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        tmp_path = None
        try:
            # Save uploaded file temporarily
            file_bytes = uploaded_file.read()
            tmp_path = os.path.join(out_dir, f"temp_{uploaded_file.name}")
            with open(tmp_path, 'wb') as f:
                f.write(file_bytes)
            
            # Extract EXIF before resize
            lat, lon = extract_gps_from_image(tmp_path)
            date, time = get_capture_date_time(tmp_path)
            
            if lat is None or lon is None:
                st.warning(f"No GPS data in {uploaded_file.name}. Skipped.")
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                continue
            
            # Resize to meet Roboflow limits
            p1, s1 = limit_resolution_to_temp(tmp_path)
            p2, s2 = progressive_resize_to_temp(p1)
            final_path = p2 or p1 or tmp_path
            scale = (s1 * s2) if p2 else (s1 if p1 else 1.0)
            
            # Read resized image and upload to S3
            with open(final_path, 'rb') as f:
                resized_bytes = f.read()
            
            s3_key = upload_to_s3_direct(resized_bytes, S3_BUCKET, uploaded_file.name)
            
            # Store metadata with s3_key
            s3_keys.append({
                's3_key': s3_key,
                'original_name': uploaded_file.name,
                'lat': lat,
                'lon': lon,
                'date': date,
                'time': time,
                'scale': scale
            })
            
            # Cleanup temp files
            for p in (p2, p1, tmp_path):
                if p and os.path.exists(p):
                    os.remove(p)
            
            upload_progress_bar.progress((idx + 1) / total_files)
            upload_progress_label.text(f"{idx + 1}/{total_files} uploaded to S3")
            
        except Exception as e:
            st.error(f"Failed to process/upload {uploaded_file.name}: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    
    st.success(f"✅ {len(s3_keys)} files uploaded to S3")
    
    # Stage 2: Process one at a time from S3
    processed_count = 0
    upload_progress_bar.progress(0)
    upload_progress_label.text(f"0/{len(s3_keys)} processed")
    progress_text = st.empty()

    for item in s3_keys:
        s3_key = item['s3_key']
        original_name = item['original_name']
        lat = item['lat']
        lon = item['lon']
        date = item['date']
        time = item['time']
        scale = item['scale']
        
        location = None
        group_key = None
        local_path = None

        try:
            progress_text.markdown(f"**Processing: {processed_count + 1}/{len(s3_keys)}** - {original_name}")
            
            # Generate presigned URL for Roboflow
            s3_url = generate_presigned_url(S3_BUCKET, s3_key, expiration=600)
            
            location = get_location_name(lat)
            group_key = (location, date)

            # Run detection from S3 URL (Roboflow pulls resized image)
            detections = run_detection_from_url(s3_url, conf_threshold, overlap_threshold)

            # Download for annotation
            local_path = os.path.join(out_dir, f"annotate_{original_name}")
            download_from_s3(S3_BUCKET, s3_key, local_path)
            
            img = cv2.imread(local_path)
            if img is None:
                raise RuntimeError(f"Could not read image {original_name}")

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
                    "image_name": original_name,
                    "seal_id": seal_idx,
                    "x_min": float(box[0]),
                    "y_min": float(box[1]),
                    "x_max": float(box[2]),
                    "y_max": float(box[3]),
                    "confidence": float(conf)
                })

            for box in detections.xyxy:
                xc = (box[0] + box[2]) / 2
                yc = (box[1] + box[3]) / 2
                dx_m = (xc - img.shape[1] / 2) * gsd
                dy_m = (yc - img.shape[0] / 2) * gsd
                lat_off = dy_m / 111320
                lon_off = dx_m / (111320 * cos(radians(lat)))
                st.session_state['grouped_coords'][group_key].append((lat + lat_off, lon + lon_off))

            st.session_state['summary_records'].append({
                "image_name": original_name,
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

            # Cleanup
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
            delete_from_s3(S3_BUCKET, s3_key)
            
            del img, labeled
            gc.collect()

            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)

            processed_count += 1
            upload_progress_bar.progress(processed_count / len(s3_keys))
            upload_progress_label.text(f"{processed_count}/{len(s3_keys)} processed")

        except Exception as e:
            st.error(f"Error processing {original_name}: {e}")
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            try:
                delete_from_s3(S3_BUCKET, s3_key)
            except:
                pass
            gc.collect()

    final_memory = process.memory_info().rss / (1024 * 1024)
    memory_used = final_memory - initial_memory

    progress_text.markdown(f"**✅ Complete! {processed_count}/{len(s3_keys)} processed**")

    st.markdown("### Annotated Images")
    for ap in st.session_state.get('annotated_paths', []):
        try:
            st.image(cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB))
        except:
            st.image(ap)

    folder_summary_records = compute_unique_counts(
        st.session_state['grouped_coords'],
        st.session_state['max_counts']
    )
    display_and_download_summary(
        st.session_state['summary_records'],
        folder_summary_records,
        st.session_state['all_detections_records']
    )

    st.markdown("### 📊 Memory Usage")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak RAM", f"{peak_memory:.1f} MB")
    with col2:
        st.metric("Final RAM", f"{final_memory:.1f} MB")
    with col3:
        st.metric("RAM Increase", f"{memory_used:.1f} MB")