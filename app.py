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
from uuid import uuid4

st.title("Pinniped Detection from Drone Imagery")
st.markdown("Upload drone images to detect seals using a YOLOv11 model (via Roboflow).")

uploaded_files = st.file_uploader("Upload Drone Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.markdown("### ⚙️ Detection Thresholds")

conf_threshold = st.slider("Confidence threshold (%)", 0, 100, 15, step=5)
overlap_threshold = st.slider("Overlap threshold (%)", 0, 100, 30, step=5)

if uploaded_files:
    import shutil

    # Create a per-run temp directory and a staged list in session state to
    # avoid holding all UploadedFile objects in memory during processing.
    if 'out_dir' not in st.session_state:
        st.session_state['out_dir'] = tempfile.mkdtemp()
    out_dir = st.session_state['out_dir']

    if 'staged_paths' not in st.session_state:
        st.session_state['staged_paths'] = []
    if 'next_index' not in st.session_state:
        st.session_state['next_index'] = 0

    # Stage uploaded files to disk immediately, then release UploadedFile refs
    for i, uploaded in enumerate(uploaded_files):
        try:
            unique_name = f"{uuid4().hex}_{uploaded.name}"
            tmp_path = os.path.join(out_dir, unique_name)
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.read())
            st.session_state['staged_paths'].append(tmp_path)
        except Exception as e:
            st.error(f"Failed to save {getattr(uploaded, 'name', str(i))}: {e}")
        finally:
            # Release the UploadedFile object reference to free memory
            try:
                uploaded.file.close()
            except Exception:
                pass
            uploaded = None

    # Free references to uploaded_files list itself so Streamlit can GC
    try:
        del uploaded_files
    except Exception:
        pass

    # Now process staged files sequentially (one at a time)
    grouped_coords, max_counts = defaultdict(list), defaultdict(int)
    all_detections_records, summary_records = [], []

    total = len(st.session_state['staged_paths'])
    progress_text = st.empty()

    for idx in range(st.session_state['next_index'], total):
        image_path = st.session_state['staged_paths'][idx]
        uploaded_name = os.path.basename(image_path)
        progress_text.markdown(f"**Progress: {idx}/{total} annotated**")

        try:
            # Preprocess: resize (these functions return temp paths)
            p1, s1 = limit_resolution_to_temp(image_path)
            p2, s2 = progressive_resize_to_temp(p1)
            scale = s1 * s2

            # Extract metadata from the original file
            lat, lon = extract_gps_from_image(image_path)
            date, time = get_capture_date_time(image_path)
            location = get_location_name(lat)

            if lat is None or lon is None:
                st.warning(f"⚠️ No GPS data in {uploaded_name}. Skipping.")
                st.session_state['next_index'] += 1
                # remove original file to free disk
                try:
                    os.remove(image_path)
                except Exception:
                    pass
                continue

            group_key = (location, date)

            # Run detection on resized image
            detections = run_detection(p2, conf_threshold, overlap_threshold)

            # Load resized image for annotation
            img = cv2.imread(p2)
            if img is None:
                st.warning(f"❌ Could not read resized image {uploaded_name}")
                raise RuntimeError("Failed to read resized image")

            gsd = compute_gsd({}, img.shape[1])

            # Annotate using detections (coordinates are relative to p2)
            annotator = sv.BoxAnnotator()
            texts = [f"seal {conf*100:.1f}%" for conf in detections.confidence]
            labeled = annotator.annotate(scene=img.copy(), detections=detections, labels=texts) \
                if "labels" in inspect.signature(annotator.annotate).parameters \
                else annotator.annotate(scene=img.copy(), detections=detections)
            st.image(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB), caption=f"✅ {uploaded_name} ({len(detections.xyxy)} seals)")

            # Save detections scaled back to original image size
            scaled_boxes = (detections.xyxy * scale) if hasattr(detections.xyxy, "__mul__") else detections.xyxy
            for seal_idx, (box, conf) in enumerate(zip(scaled_boxes, detections.confidence), start=1):
                all_detections_records.append({
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
                grouped_coords[group_key].append(((lat + lat_off, lon + lon_off), gsd, time))

            summary_records.append({
                "image_name": uploaded_name,
                "pinniped_count": len(detections.xyxy),
                "latitude": lat,
                "longitude": lon,
                "location": location,
                "date": date,
                "time": time
            })

            # Done with this image: increment index and remove original file to free disk
            st.session_state['next_index'] += 1
            try:
                os.remove(image_path)
            except Exception:
                pass

            # Free memory
            del img
            gc.collect()

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_name}: {e}")
            # ensure we still advance to next image to avoid stuck loops
            st.session_state['next_index'] += 1
            try:
                os.remove(image_path)
            except Exception:
                pass
            gc.collect()
            continue

    # Final summary and cleanup
    progress_text.markdown(f"**✅ Complete! {st.session_state.get('next_index', 0)}/{total} annotated**")
    folder_summary_records = compute_unique_counts(grouped_coords, max_counts)
    display_and_download_summary(summary_records, folder_summary_records, all_detections_records)
