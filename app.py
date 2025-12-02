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
<<<<<<< HEAD

from scripts.detection import run_detection_from_url
=======
from uuid import uuid4

from scripts.detection import run_detection
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary
<<<<<<< HEAD
from scripts.s3_utils import upload_to_s3_direct, generate_presigned_url, delete_from_s3, download_from_s3

try:
    S3_BUCKET = st.secrets["S3_BUCKET_NAME"]
except:
    S3_BUCKET = os.getenv("S3_BUCKET_NAME", "pinniped-drone-uploads")
=======
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede

st.title("Pinniped Detection from Drone Imagery")

st.markdown("### ⚙️ Detection Thresholds")
conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap / NMS threshold (%)", 0, 100, 30, step=5)

st.markdown("Upload drone images to detect seals using Roboflow serverless inference.")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

<<<<<<< HEAD
if uploaded_files:
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)
=======
upload_progress_bar = st.empty()
upload_progress_label = st.empty()

if uploaded_files:
    # Track RAM usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede
    peak_memory = initial_memory
    
    if 'out_dir' not in st.session_state:
        st.session_state['out_dir'] = tempfile.mkdtemp()
    out_dir = st.session_state['out_dir']

<<<<<<< HEAD
    # Initialize session state only once
    if 'processing_started' not in st.session_state:
        st.session_state['processing_started'] = True
        st.session_state['annotated_paths'] = []
        st.session_state['all_detections_records'] = []
        st.session_state['summary_records'] = []
        st.session_state['grouped_coords'] = defaultdict(list)
        st.session_state['max_counts'] = defaultdict(int)
        st.session_state['processed_files'] = []

    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    progress_label = st.empty()
    status_text = st.empty()

    # Process one file at a time immediately
    for idx, uploaded_file in enumerate(uploaded_files):
        # Skip if already processed (prevents reprocessing on reruns)
        if uploaded_file.name in st.session_state['processed_files']:
            continue

        tmp_path = None
        local_path = None
        s3_key = None

        try:
            status_text.markdown(f"**Processing {idx + 1}/{total_files}**: {uploaded_file.name}")

            # Write to disk and immediately free memory
            tmp_path = os.path.join(out_dir, f"tmp_{idx}_{uploaded_file.name}")
            with open(tmp_path, 'wb') as f:
                # Read in chunks to avoid loading entire file in memory
                while True:
                    chunk = uploaded_file.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Extract EXIF
            lat, lon = extract_gps_from_image(tmp_path)
            date, time = get_capture_date_time(tmp_path)

            if lat is None or lon is None:
                st.warning(f"No GPS in {uploaded_file.name}. Skipped.")
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                st.session_state['processed_files'].append(uploaded_file.name)
                continue

            # Resize and upload to S3
            p1, s1 = limit_resolution_to_temp(tmp_path)
            p2, s2 = progressive_resize_to_temp(p1)
            final_path = p2 or p1 or tmp_path
            scale = (s1 * s2) if p2 else (s1 if p1 else 1.0)

            with open(final_path, 'rb') as f:
                resized_bytes = f.read()
            
            s3_key = upload_to_s3_direct(resized_bytes, S3_BUCKET, f"upload_{idx}_{uploaded_file.name}")
            del resized_bytes

            # Cleanup local temps
            for p in (p2, p1, tmp_path):
                if p and os.path.exists(p):
                    os.remove(p)
            gc.collect()

            # Generate presigned URL and run detection
            s3_url = generate_presigned_url(S3_BUCKET, s3_key, expiration=600)
            location = get_location_name(lat)
            group_key = (location, date)

            detections = run_detection_from_url(s3_url, conf_threshold, overlap_threshold)

            # Download for annotation only
            local_path = os.path.join(out_dir, f"ann_{idx}.jpg")
            download_from_s3(S3_BUCKET, s3_key, local_path)

            img = cv2.imread(local_path)
            if img is None:
                raise RuntimeError(f"Could not read {uploaded_file.name}")
=======
    st.session_state['annotated_paths'] = []
    st.session_state['all_detections_records'] = []
    st.session_state['summary_records'] = []
    st.session_state['grouped_coords'] = defaultdict(list)
    st.session_state['max_counts'] = defaultdict(int)

    total_files = len(uploaded_files)
    processed_count = 0

    upload_progress_bar.progress(0)
    upload_progress_label.text(f"0/{total_files} processed")
    progress_text = st.empty()

    for uploaded in uploaded_files:
        uploaded_name = uploaded.name
        unique_name = f"{uuid4().hex}_{uploaded_name}"
        tmp_path = os.path.join(out_dir, unique_name)
        
        location = None
        group_key = None

        try:
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.read())

            lat, lon = extract_gps_from_image(tmp_path)
            date, time = get_capture_date_time(tmp_path)

            progress_text.markdown(f"**Processing: {processed_count + 1}/{total_files}**")

            p1, s1 = limit_resolution_to_temp(tmp_path)
            p2, s2 = progressive_resize_to_temp(p1)
            candidate_path = p2 or p1 or tmp_path
            scale = (s1 * s2) if p2 else (s1 if p1 else 1.0)

            if lat is None or lon is None:
                st.warning(f"No GPS data in {uploaded_name}. Skipped.")
                for p in (p2, p1, tmp_path):
                    if p and os.path.exists(p):
                        os.remove(p)
                continue

            location = get_location_name(lat)
            group_key = (location, date)

            detections = run_detection(candidate_path, conf_threshold, overlap_threshold)

            img = cv2.imread(candidate_path)
            if img is None:
                raise RuntimeError(f"Could not read image {uploaded_name}")
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede

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

            scaled_boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale] for b in detections.xyxy]
<<<<<<< HEAD

            for seal_idx, (box, conf) in enumerate(zip(scaled_boxes, detections.confidence), start=1):
                st.session_state['all_detections_records'].append({
                    "image_name": uploaded_file.name,
=======
            for seal_idx, (box, conf) in enumerate(zip(scaled_boxes, detections.confidence), start=1):
                st.session_state['all_detections_records'].append({
                    "image_name": uploaded_name,
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede
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
<<<<<<< HEAD
                "image_name": uploaded_file.name,
                "pinniped_count": len(scaled_boxes),
=======
                "image_name": uploaded_name,
                "pinniped_count": len(detections.xyxy),
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede
                "latitude": lat,
                "longitude": lon,
                "location": location,
                "date": date,
                "time": time
            })

            key = (location, date)
            current_max = st.session_state['max_counts'].get(key, 0)
<<<<<<< HEAD
            st.session_state['max_counts'][key] = max(current_max, len(scaled_boxes))

            # Cleanup
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
            delete_from_s3(S3_BUCKET, s3_key)

            del img, labeled
            gc.collect()

            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)

            st.session_state['processed_files'].append(uploaded_file.name)
            progress_bar.progress((idx + 1) / total_files)
            progress_label.text(f"{idx + 1}/{total_files} complete")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            for p in (local_path, tmp_path):
=======
            st.session_state['max_counts'][key] = max(current_max, len(detections.xyxy))

            for p in (p2, p1, tmp_path):
                if p and os.path.exists(p):
                    os.remove(p)
            
            del img, labeled
            gc.collect()

            # Track peak memory
            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)

            processed_count += 1
            upload_progress_bar.progress(processed_count / total_files)
            upload_progress_label.text(f"{processed_count}/{total_files} processed")

        except Exception as e:
            st.error(f"Error processing {uploaded_name}: {e}")
            for p in (locals().get('candidate_path'), locals().get('p2'), locals().get('p1'), tmp_path):
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass
<<<<<<< HEAD
            if s3_key:
                try:
                    delete_from_s3(S3_BUCKET, s3_key)
                except:
                    pass
            gc.collect()

    final_memory = process.memory_info().rss / (1024 * 1024)
    memory_used = final_memory - initial_memory

    status_text.markdown(f"**✅ Complete! {len(st.session_state['processed_files'])}/{total_files} processed**")
=======
            gc.collect()

    # Final memory stats
    final_memory = process.memory_info().rss / (1024 * 1024)
    memory_used = final_memory - initial_memory

    progress_text.markdown(f"**✅ Complete! {processed_count}/{total_files} processed**")
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede

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
<<<<<<< HEAD
=======

# Display RAM usage
>>>>>>> 63c828a4dc9aefaa13635e493499c2bc297b9ede
    st.markdown("### 📊 Memory Usage")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak RAM", f"{peak_memory:.1f} MB")
    with col2:
        st.metric("Final RAM", f"{final_memory:.1f} MB")
    with col3:
        st.metric("RAM Increase", f"{memory_used:.1f} MB")