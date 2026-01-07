# Pinniped Drone Detection App - Technical Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Understanding Detection Thresholds](#understanding-detection-thresholds)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Pinniped Drone Detection App is a Streamlit-based web application that uses a custom-trained YOLO v11 computer vision model to automatically detect and count pinnipeds (seals and sea lions) in drone imagery. The application processes images from Dropbox, performs inference using Roboflow's hosted model, and generates detailed CSV reports with detection counts and geographic clustering.

### Key Features
- **Automated Detection**: YOLO v11 model with 94.4% precision and 72.7% recall
- **Dropbox Integration**: Direct ZIP file processing from shared Dropbox links
- **S3 Storage**: Intermediate cloud storage for scalable image processing
- **GPS-Based Clustering**: DBSCAN algorithm to estimate unique pinniped counts across multiple images
- **Adjustable Thresholds**: User-configurable confidence and NMS thresholds
- **ODK Integration**: Direct upload link to Open Data Kit for field data collection

### Technology Stack
- **Frontend**: Streamlit
- **Computer Vision**: YOLO v11 (via Roboflow), OpenCV, Supervision
- **Cloud Storage**: AWS S3
- **Clustering**: scikit-learn (DBSCAN), pyproj
- **Image Processing**: Pillow, piexif

---

## Architecture

### System Components

```
┌─────────────────┐
│  Dropbox ZIP    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AWS S3        │ ◄──── Session-based temp storage
│  (Staging)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit App  │
│   (Processing)  │
├─────────────────┤
│ • EXIF Extract  │
│ • Image Resize  │
│ • ML Inference  │
│ • Clustering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CSV Output    │
│   • Per-Image   │
│   • Per-Location│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   ODK Upload    │
└─────────────────┘
```

### File Structure

```
pinniped_drone_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System packages
├── .streamlit/
│   └── secrets.toml         # Secret credentials
├── scripts/
│   ├── config.py            # Configuration constants
│   ├── detection.py         # Roboflow inference & NMS
│   ├── exif_utils.py        # GPS/EXIF extraction
│   ├── image_utils.py       # Image scaling & GSD
│   ├── clustering.py        # DBSCAN unique counting
│   ├── s3_utils.py          # AWS S3 operations
│   └── summaries.py         # CSV generation & display
└── test images/             # Sample test data
```

---

## Data Flow

### Step-by-Step Processing Pipeline

#### 1. **Input Stage**
```
User Input → Dropbox ZIP URL
           → Validate URL format
           → Convert to direct download link
```

#### 2. **Download & Extract**
```
Stream ZIP from Dropbox (max 2 GB)
  → Extract images (.jpg, .jpeg, .png)
  → Upload each to S3 (session-based folder)
  → Clean up local ZIP from memory
```

#### 3. **Per-Image Processing Loop**
For each image in S3:

**a. Download & EXIF Extraction**
```python
download_from_s3(bucket, s3_key, local_path)
  → extract_gps_from_image()       # lat, lon
  → get_capture_date_time()        # date, time
  → get_location_name(lat)         # geographic location
```

**b. Image Preparation**
```python
limit_resolution_to_temp(image)   # Max 4M pixels, 15 MB
  → progressive_resize_to_temp()   # Further downsampling if needed
  → compute_gsd(img_width)         # Ground Sample Distance
```

**c. ML Inference**
```python
run_detection_from_local(image_path, conf_thresh, overlap_thresh)
  → Roboflow API call
  → parse_roboflow_detections()    # Convert JSON to Supervision format
  → Custom NMS filtering           # Remove duplicate boxes
  → Return: Detections object (boxes, confidences, class_ids)
```

**d. Coordinate Projection**
```python
For each detection box:
  1. Get box center in pixels (xc, yc)
  2. Convert to meters from image center using GSD
  3. Project to geographic coordinates:
     lat_offset = dy_m / 111320
     lon_offset = dx_m / (111320 * cos(radians(lat)))
  4. Store: (lat + lat_offset, lon + lon_offset)
```

**e. Annotation & Storage**
```python
annotate_image(detections)        # Add bounding boxes
  → Save to temp file
  → Store in session_state
  → Cleanup resized temps
```

#### 4. **Clustering Stage**
```python
compute_unique_counts(grouped_coords, max_counts)
  → Group detections by (location, date)
  → Project to Web Mercator (EPSG:3857)
  → Calculate adaptive DBSCAN epsilon:
      - Compute nearest-neighbor distances
      - eps = clip(median * scale_factor, eps_min, eps_max)
  → Run DBSCAN clustering
  → Count unique clusters
  → Enforce lower bound: max(cluster_count, max_single_image_count)
```

#### 5. **Output Generation**
```python
display_and_download_summary()
  → Generate per-image CSV (filename, count, GPS, date/time)
  → Generate per-location CSV (location, date, unique_count)
  → Generate all-detections CSV (bounding box coordinates, confidence)
  → Display annotated images
  → Provide download buttons
```

#### 6. **Cleanup**
```
Delete S3 session folder
Clean local temp files
Display memory usage stats
```

---

## Module Documentation

### `config.py`
**Purpose**: Centralized configuration constants

**Constants**:
- `MAX_PIXELS = 4_000_000` - Maximum image resolution for Roboflow
- `MAX_SIZE_MB = 15` - Maximum file size for Roboflow
- `MIN_SCALE_PERCENT = 10` - Minimum allowed scaling
- `CAMERA_SENSOR_WIDTHS` - Dict of camera models and sensor widths (mm)

---

### `exif_utils.py`
**Purpose**: Extract EXIF metadata and GPS coordinates from images

#### `extract_gps_from_image(path: str) -> tuple[float, float] | tuple[None, None]`
Extracts GPS coordinates from image EXIF data.

**Returns**: `(latitude, longitude)` or `(None, None)` if not found

**Example**:
```python
lat, lon = extract_gps_from_image("image.jpg")
# (49.2827, -123.1207)
```

#### `get_capture_date_time(path: str) -> tuple[str, str]`
Extracts capture date and time from EXIF.

**Returns**: `(date, time)` as strings or `("Unknown", "Unknown")`

**Example**:
```python
date, time = get_capture_date_time("image.jpg")
# ("2024-08-15", "14:32:10")
```

#### `get_location_name(lat: float) -> str`
Maps latitude to predefined location names.

**Logic**:
- `lat > 50.0` → "Campbell River"
- `49.0 < lat ≤ 50.0` → "Nanaimo"
- `lat ≤ 49.0` → "Cowichan"

---

### `image_utils.py`
**Purpose**: Image scaling and Ground Sample Distance calculations

#### `compute_gsd(img_width: int) -> float`
Calculates Ground Sample Distance (meters per pixel).

**Assumptions**:
- Drone altitude: 50m
- Camera: DJI Mavic 3 (sensor width: 13.2mm, focal length: 24mm)

**Formula**:
```
GSD = (altitude × sensor_width) / (focal_length × img_width)
```

**Returns**: GSD in meters/pixel

---

#### `limit_resolution_to_temp(image_path: str) -> tuple[str | None, float]`
Downscales image if it exceeds Roboflow limits.

**Limits**:
- Max pixels: 4,000,000
- Max file size: 15 MB

**Returns**: `(temp_path, scale_factor)` or `(None, 1.0)` if no resize needed

---

#### `progressive_resize_to_temp(path: str | None) -> tuple[str | None, float]`
Further reduces image size if still too large.

**Strategy**: Iteratively scale by 0.9× until size < 15 MB

**Returns**: `(temp_path, scale_factor)` or `(None, 1.0)`

---

### `detection.py`
**Purpose**: ML inference and Non-Maximum Suppression

#### `run_detection_from_local(image_path: str, confidence_percent: int, overlap_percent: int) -> sv.Detections`
Runs Roboflow inference from a local file path and applies custom NMS. Not currently in use, but retained for reference.

**Parameters**:
- `confidence_percent`: 0-100, threshold for valid detections
- `overlap_percent`: 0-100, IoU threshold for NMS

**Process**:
1. Read image with OpenCV
2. Call Roboflow API with `MODEL_ID = "pinnipeds-drone-imagery/18"`
3. Parse JSON response to extract boxes, confidences, class IDs
4. Apply custom NMS filtering
5. Return `supervision.Detections` object

**Returns**: Detections with attributes:
- `xyxy`: Bounding boxes (N×4 array)
- `confidence`: Confidence scores (N array)
- `class_id`: Class IDs (N array)

---

### `clustering.py`
**Purpose**: Spatial clustering to estimate unique pinniped counts

#### `compute_unique_counts(grouped_coords: dict, max_counts: dict, eps_min=3.0, eps_max=20.0, scale_factor=1.6) -> list[dict]`

**Algorithm**:
1. **Group** detections by `(location, date)`
2. **Project** lat/lon to Web Mercator (EPSG:3857) for Euclidean distance
3. **Adaptive epsilon**:
   - Compute k=2 nearest neighbors
   - `eps = clip(median(distances) × scale_factor, eps_min, eps_max)`
4. **DBSCAN** clustering with `min_samples=1`
5. **Post-merge**: Merge clusters closer than `eps/2`
6. **Lower bound**: `unique_count = max(cluster_count, max_single_image_count)`

**Parameters**:
- `grouped_coords`: `{(location, date): [(lat, lon), ...]}`
- `max_counts`: `{(location, date): max_count_in_single_image}`
- `eps_min`: Minimum cluster radius (meters)
- `eps_max`: Maximum cluster radius (meters)
- `scale_factor`: Multiplier for adaptive epsilon

**Returns**: List of dicts with keys:
- `survey_location`: Location name
- `date`: Survey date
- `unique_count`: Estimated unique pinnipeds

---

### `s3_utils.py`
**Purpose**: AWS S3 operations for intermediate storage

#### `upload_to_s3_direct(file_data: bytes, bucket_name: str, s3_key: str) -> bool`
Uploads file data directly to S3.

#### `download_from_s3(bucket_name: str, s3_key: str, local_path: str) -> bool`
Downloads file from S3 to local path.

#### `list_s3_files(bucket_name: str, prefix: str) -> list[str]`
Lists all files in S3 with given prefix.

#### `delete_s3_folder(bucket_name: str, folder_prefix: str) -> bool`
Deletes all objects in S3 folder.

#### `clean_entire_dropbox_extracts_folder(bucket_name: str) -> bool`
Cleans the entire `dropbox_extracts/` folder.

**S3 Structure**:
```
s3://bucket-name/
└── dropbox_extracts/
    ├── {session_id_1}/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── {session_id_2}/
        └── image3.jpg
```

---

### `summaries.py`
**Purpose**: Generate and display CSV summaries

#### `display_and_download_summary(summary_records: list, folder_summary_records: list, all_detections_records: list) -> None`

Displays three CSV download buttons:

1. **Per-Image Summary**
   - Columns: `image_name`, `pinniped_count`, `latitude`, `longitude`, `location`, `date`, `time`
   
2. **Per-Location Summary**
   - Columns: `survey_location`, `date`, `unique_count`
   
3. **All Detections**
   - Columns: `image_name`, `seal_id`, `x_min`, `y_min`, `x_max`, `y_max`, `confidence`

---

## Configuration

### Environment Variables / Secrets

Required in `.streamlit/secrets.toml` or environment:

```toml
# Roboflow API
ROBOFLOW_API_KEY = "your_api_key_here"

# AWS S3
AWS_ACCESS_KEY_ID = "your_access_key"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
AWS_REGION = "us-west-1"
S3_BUCKET_NAME = "your-bucket-name"

# ODK Integration
ODK_UPLOAD_URL = "https://odk.example.com/upload-link"
```

### Deployment on Streamlit Cloud

1. **Fork/Clone** repository
2. **Connect** to Streamlit Cloud
3. **Add Secrets** in app settings (same format as above)
4. **Deploy** - app will auto-detect secrets

### AWS S3 Setup

1. **Create S3 bucket** (e.g., `drone-images-temp-uploads`)
2. **Set IAM permissions**:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": [
         "s3:PutObject",
         "s3:GetObject",
         "s3:DeleteObject",
         "s3:ListBucket"
       ],
       "Resource": [
         "arn:aws:s3:::your-bucket-name",
         "arn:aws:s3:::your-bucket-name/*"
       ]
     }]
   }
   ```
3. **Optional**: Set up Lambda for weekly cleanup (see below)

### Lambda Cleanup Schedule (Optional)

For automatic S3 cleanup every Sunday at 2 AM PST:

**Lambda Function**:
```python
import boto3
import os

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='dropbox_extracts/')
    
    for page in pages:
        if 'Contents' in page:
            delete_objects = [{'Key': obj['Key']} for obj in page['Contents']]
            if delete_objects:
                s3_client.delete_objects(Bucket=bucket_name, Delete={'Objects': delete_objects})
    
    return {'statusCode': 200}
```

**EventBridge Schedule**: `cron(0 10 ? * SUN *)` (10 AM UTC = 2 AM PST)

---

## API Reference

### Roboflow API

**Endpoint**: `https://serverless.roboflow.com`

**Model**: `pinnipeds-drone-imagery/18`

**Request**:
```python
client.infer(image_path, model_id="pinnipeds-drone-imagery/18")
```

**Response Format**:
```json
{
  "predictions": [
    {
      "x": 512.5,
      "y": 384.0,
      "width": 64.0,
      "height": 48.0,
      "confidence": 0.92,
      "class": "pinniped",
      "class_id": 0
    }
  ],
  "image": {"width": 1024, "height": 768}
}
```

---

## Understanding Detection Thresholds

### Confidence Slider

**What it means**: The confidence slider (0-100%) controls the minimum probability threshold for accepting a detection. The model outputs a confidence score between 0-1 for each detected object, representing how certain it is that the detection is actually a pinniped.

**How it works**:
- A confidence of 95% means "only show me detections the model is at least 95% confident about"
- Lower values (10-20%) accept more detections, including uncertain ones
- Higher values (30-50%) filter out uncertain predictions, keeping only the most reliable detections

**When to adjust**:
- **Increase confidence** (→ 30-40%):
  - When you're getting too many false positives (detecting rocks, water, shadows as pinnipeds)
  - When working with poor image quality or unusual lighting
  - When accuracy is more important than finding every animal
  
- **Decrease confidence** (→ 10-15%):
  - When you're missing detections (the model isn't finding all the pinnipeds)
  - When working with high-quality drone imagery with good lighting
  - When finding every animal is more important than eliminating false positives
  - In crowded scenes where some pinnipeds may be partially obscured

**Starting recommendation**: Begin at 15% confidence. This provides a good balance for typical drone surveys. Adjust from there based on your image quality and results.

---

### Overlap (NMS) Slider

**What it means**: The overlap slider (0-100%) controls the Non-Maximum Suppression (NMS) threshold, which removes duplicate or overlapping bounding boxes. When the model detects the same pinniped multiple times or detects overlapping boxes around the same animal, NMS merges them into a single detection.

**How it works**:
- NMS measures the Intersection-over-Union (IoU) between bounding boxes
- An overlap threshold of 30% means "merge any two boxes that overlap by more than 30%"
- Lower values (10-20%) are more aggressive at removing duplicates
- Higher values (50-80%) are less aggressive and keep more overlapping boxes

**When to adjust**:
- **Decrease overlap** (→ 10-20%):
  - When you're seeing duplicate detections of the same pinniped (counted twice)
  - When boxes are stacked or heavily overlapping
  - When you need cleaner, non-redundant detections
  
- **Increase overlap** (→ 50-70%):
  - When you're losing detections because NMS is merging legitimate separate pinnipeds
  - When pinnipeds are tightly clustered or huddled together
  - When you want to detect all individuals, even if some boxes overlap slightly

**Starting recommendation**: Begin at 30% overlap. This removes most duplicates while preserving detections of closely-spaced animals. Adjust only if you notice specific issues with your results.

---

### Typical Tuning Scenarios

**Scenario 1: Too many false positives (rocks, foam, shadows)**
- Increase confidence: 15% → 25-35%
- Keep overlap at 30%

**Scenario 2: Missing detections (animals not detected)**
- Decrease confidence: 15% → 10%
- Decrease overlap: 30% → 20% (in case the model generates multiple weak predictions)

**Scenario 3: Duplicate detections of the same animal**
- Keep confidence the same
- Decrease overlap: 30% → 15-20%

**Scenario 4: Clustered pinnipeds not being distinguished**
- Increase overlap: 30% → 50%
- May need to also decrease confidence slightly to ensure all individuals are detected

**Scenario 5: High-altitude survey (small pinnipeds in image)**
- Decrease confidence: 15% → 8-12%
- Decrease overlap: 30% → 15-20%
- The model is less certain at high altitudes, so more lenient thresholds help capture all detections

---

### Iterative Tuning Workflow

1. **Start with defaults**: Confidence 15%, Overlap 30%
2. **Process a small batch** of representative images (5-10 images from different conditions)
3. **Review results**:
   - Count obvious false positives or duplicates
   - Identify any missed animals
   - Note the image conditions (altitude, lighting, density)
4. **Adjust one parameter** based on observed issues
5. **Reprocess** the same batch to see impact
6. **Finalize settings** once results look good, then process full dataset

---

## Troubleshooting

### Common Issues

#### 1. **"No GPS in image. Skipped."**
**Cause**: Image lacks GPS EXIF metadata  
**Solution**: Ensure drone GPS was enabled during capture. Images without GPS are automatically skipped.

---

#### 2. **"ZIP file is too large"**
**Cause**: ZIP exceeds 2 GB limit  
**Solution**: Split images into multiple ZIP files < 2 GB each, process separately.

---

#### 3. **S3 Upload Failures**
**Cause**: Invalid AWS credentials or permissions  
**Solution**: 
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- Check IAM policy includes `s3:PutObject` permission
- Ensure bucket name is correct

---

#### 4. **Low Detection Accuracy**
**Cause**: Suboptimal threshold settings  
**Solution**:
- **Too many false positives**: Increase confidence threshold (15% → 25%)
- **Missing detections**: Decrease confidence threshold (15% → 10%)
- **Duplicate boxes**: Decrease overlap threshold (30% → 20%)

---

#### 5. **Memory Issues on Streamlit Cloud**
**Cause**: Large image processing loads  
**Solution**:
- App automatically resizes images to stay within limits
- S3 staging ensures only one image in memory at a time
- Garbage collection runs after each image

---

#### 6. **Clustering Issues (Unique Counts Too High/Low)**
**Cause**: Inappropriate DBSCAN parameters  
**Solution**: Adjust in `clustering.py`:
```python
compute_unique_counts(
    grouped_coords, 
    max_counts,
    eps_min=3.0,      # Increase to merge more
    eps_max=20.0,     # Decrease to split more
    scale_factor=1.6  # Adjust adaptive epsilon sensitivity
)
```

---

### Debugging Tips

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Session State
```python
st.write("Session State:", st.session_state)
```

#### Inspect Detections
```python
st.write("Detections:", detections.xyxy)
st.write("Confidences:", detections.confidence)
```

#### Monitor Memory
```python
import psutil
process = psutil.Process()
st.write(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

---

## Performance Metrics

### Model Performance (Test Set)
- **Precision**: 94.4%
- **Recall**: 72.7%
- **Accuracy**: 80.0%
- **F1 Score**: 82.1%
- **AUC-ROC**: 0.83

### Processing Speed
- **Average time per image**: 3-5 seconds
- **Breakdown**:
  - Download from S3: ~0.5s
  - EXIF extraction: ~0.1s
  - Image resize: ~0.3s
  - ML inference: 2-3s
  - Annotation: ~0.5s

### Scalability
- **Max ZIP size**: 2 GB
- **Max images per session**: ~20-30 (if each image is >30 GB)
- **Concurrent users**: Handled via Streamlit Cloud autoscaling

---

## Future Enhancements

### Scoped Features
- Support for Google Drive / OneDrive links
- Species classification (harbor seals vs. sea lions)
- Interactive dashboard for visualization of detections
- Batch processing queue for large datasets

### Known Limitations
- Only processes `.jpg`, `.jpeg`, `.png` formats
- Requires GPS metadata (images without GPS are skipped)
- 2 GB ZIP file limit
- Single model version (no A/B testing)

---

## Support & Contact

For issues, questions, or contributions:
- **GitHub**: https://github.com/riyaeliza123/pinniped_drone_app
- **Live App**: https://riyaeliza123-pinniped-drone-app.streamlit.app/

---

**Last Updated**: December 2025  
**Version**: 1.0  
**License**: MIT
