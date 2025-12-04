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
import requests
import zipfile
import io

from scripts.detection import run_detection_from_local
from scripts.exif_utils import extract_gps_from_image, get_capture_date_time, get_location_name
from scripts.image_utils import limit_resolution_to_temp, progressive_resize_to_temp, compute_gsd
from scripts.clustering import compute_unique_counts
from scripts.summaries import display_and_download_summary
from scripts.s3_utils import (
    upload_to_s3_direct, 
    download_from_s3, 
    list_s3_files,
    delete_s3_folder
)

# Get S3 bucket name
try:
    S3_BUCKET = st.secrets["S3_BUCKET_NAME"]
except:
    S3_BUCKET = os.getenv("S3_BUCKET_NAME")
    if not S3_BUCKET:
        st.error("S3_BUCKET_NAME not configured. Please add it to .streamlit/secrets.toml")
        st.stop()

st.title("Pinniped Detection from Drone Imagery of Log Booms")

st.markdown("### ⚙️ Detection Thresholds")
conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap / NMS threshold (%)", 0, 100, 30, step=5)

st.markdown("### 📦 Dropbox ZIP Link")
st.info("""
**How to share via Dropbox:**
1. Upload your images folder as a ZIP file to Dropbox. The size of the ZIP **must not exceed 2 GB**.
2. Right-click the ZIP → Share → Copy link.
3. Paste the link below and press Enter.
4. Click "Download and Process from Dropbox".

**Note:** The ZIP will be downloaded to S3, extracted, and processed from there.
""")

dropbox_url = st.text_input("Dropbox ZIP file link:", value="")

if not dropbox_url:
    st.info("👆 Enter a Dropbox ZIP link to begin")
    st.stop()

# Convert Dropbox share link to direct download link
if 'dropbox.com' not in dropbox_url:
    st.error("Please provide a valid Dropbox link")
    st.stop()

direct_url = dropbox_url.replace('dl=0', 'dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')

# Process Dropbox ZIP
image_files = []
s3_folder_key = None

if st.button("Download and Process from Dropbox"):
    with st.spinner("Downloading ZIP file from Dropbox..."):
        try:
            import uuid
            session_id = uuid.uuid4().hex
            s3_folder_key = f"dropbox_extracts/{session_id}/"
            
            # Download ZIP file in chunks
            response = requests.get(direct_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Check file size limit (2 GB)
            MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
            if total_size > MAX_SIZE_BYTES:
                st.error(f"❌ ZIP file is too large: {total_size / (1024*1024*1024):.2f} GB")
                st.warning(f"📦 Maximum allowed size: {MAX_SIZE_BYTES / (1024*1024*1024):.2f} GB")
                st.info("Please split your images into smaller ZIP files (< 2 GB each) and process them separately.")
                if s3_folder_key:
                    delete_s3_folder(S3_BUCKET, s3_folder_key)
                st.stop()
            
            downloaded = 0
            
            download_progress = st.progress(0)
            download_status = st.empty()
            
            # Download to memory
            zip_bytes = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_bytes.write(chunk)
                    downloaded += len(chunk)
                    
                    # Check size during download
                    if downloaded > MAX_SIZE_BYTES:
                        st.error(f"❌ ZIP file exceeded 2 GB limit during download")
                        st.info("Please split your images into smaller ZIP files (< 2 GB each) and process them separately.")
                        if s3_folder_key:
                            delete_s3_folder(S3_BUCKET, s3_folder_key)
                        st.stop()
                    
                    if total_size > 0:
                        progress = downloaded / total_size
                        download_progress.progress(min(progress, 1.0))
                        download_status.text(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
            
            download_status.text("Download complete! Uploading to S3 and extracting...")
            
            # Reset BytesIO position
            zip_bytes.seek(0)
            
            # Extract and upload each file to S3
            extract_count = 0
            with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                extract_progress = st.progress(0)
                extract_status = st.empty()
                
                for idx, file_name in enumerate(file_list):
                    # Skip directories and hidden files
                    if file_name.endswith('/') or '/__MACOSX/' in file_name or file_name.startswith('.'):
                        continue
                    
                    # Only process image files
                    if not any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        continue
                    
                    extract_status.text(f"Extracting: {file_name}")
                    
                    # Read file from ZIP
                    file_data = zip_ref.read(file_name)
                    
                    # Upload to S3
                    s3_key = f"{s3_folder_key}{os.path.basename(file_name)}"
                    upload_to_s3_direct(file_data, S3_BUCKET, s3_key)
                    extract_count += 1
                    
                    extract_progress.progress((idx + 1) / len(file_list))
            
            extract_status.empty()
            extract_progress.empty()
            download_status.empty()
            download_progress.empty()
            
            if extract_count == 0:
                st.error("No images found in ZIP file")
                delete_s3_folder(S3_BUCKET, s3_folder_key)
                st.stop()
            
            # List all uploaded files from S3
            image_files = list_s3_files(S3_BUCKET, s3_folder_key)
            
            st.success(f"Found {len(image_files)} images uploaded to S3")
            
            # Store S3 folder key in session state for cleanup
            st.session_state['s3_folder_key'] = s3_folder_key
            
        except requests.RequestException as e:
            st.error(f"Failed to download from Dropbox: {e}")
            st.info("Make sure the link is a direct download link (ends with dl=1)")
            if s3_folder_key:
                delete_s3_folder(S3_BUCKET, s3_folder_key)
            st.stop()
        except zipfile.BadZipFile:
            st.error("Downloaded file is not a valid ZIP file")
            if s3_folder_key:
                delete_s3_folder(S3_BUCKET, s3_folder_key)
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.text(traceback.format_exc())
            if s3_folder_key:
                delete_s3_folder(S3_BUCKET, s3_folder_key)
            st.stop()
else:
    st.stop()

# Processing logic
if image_files:
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

    for idx, image_source in enumerate(image_files):
        p1 = None
        p2 = None
        tmp_path = None
        
        try:
            # Download from S3
            filename = os.path.basename(image_source)
            tmp_path = os.path.join(out_dir, f"s3_{idx}_{filename}")
            
            status_text.markdown(f"**Downloading {idx + 1}/{total_files}**: {filename}")
            download_from_s3(S3_BUCKET, image_source, tmp_path)
            
            image_path = tmp_path
            
            status_text.markdown(f"**Processing {idx + 1}/{total_files}**: {filename}")

            # Extract EXIF
            lat, lon = extract_gps_from_image(image_path)
            date, time = get_capture_date_time(image_path)

            if lat is None or lon is None:
                st.warning(f"No GPS in {filename}. Skipped.")
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
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

            # NOW cleanup resize temps and downloaded file (after reading)
            for p in (p2, p1, tmp_path):
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
            st.error(f"Error processing {filename if 'filename' in locals() else 'image'}: {e}")
            import traceback
            st.text(traceback.format_exc())
            
            # Cleanup on error
            for p in (p2, p1, tmp_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass
            gc.collect()

    final_memory = process.memory_info().rss / (1024 * 1024)
    memory_used = final_memory - initial_memory

    status_text.markdown(f"**✅ Complete! {len(st.session_state['summary_records'])}/{total_files} processed**")

    # Cleanup S3 folder if it exists
    if 's3_folder_key' in st.session_state:
        try:
            with st.spinner("Cleaning up S3..."):
                delete_s3_folder(S3_BUCKET, st.session_state['s3_folder_key'])
            del st.session_state['s3_folder_key']
        except Exception as e:
            st.warning(f"Could not cleanup S3: {e}")

    st.markdown("### 📊 Memory Usage")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak RAM", f"{peak_memory:.1f} MB")
    with col2:
        st.metric("Final RAM", f"{final_memory:.1f} MB")
    with col3:
        st.metric("RAM Increase", f"{memory_used:.1f} MB")

    st.markdown("### 📷 Annotated Images")
    for idx, ap in enumerate(st.session_state.get('annotated_paths', [])):
        try:
            if idx < len(st.session_state['summary_records']):
                filename = st.session_state['summary_records'][idx]['image_name']
            else:
                filename = f"Image {idx + 1}"
            
            st.image(cv2.cvtColor(cv2.imread(ap), cv2.COLOR_BGR2RGB))
            st.markdown(f"<p style='text-align: center; color: gray; font-size: 12px; margin-top: -15px;'>Img: {filename}</p>", unsafe_allow_html=True)
        except:
            st.image(ap)
            if idx < len(st.session_state['summary_records']):
                filename = st.session_state['summary_records'][idx]['image_name']
                st.markdown(f"<p style='text-align: center; color: gray; font-size: 12px; margin-top: -15px;'>Img: {filename}</p>", unsafe_allow_html=True)

    folder_summary_records = compute_unique_counts(
        st.session_state['grouped_coords'],
        st.session_state['max_counts']
    )
    display_and_download_summary(
        st.session_state['summary_records'],
        folder_summary_records,
        st.session_state['all_detections_records']
    )