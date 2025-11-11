# Pinniped census app using drone images

## About
A Computer Vision (YOLO v11) model trained to automatically count pinnipeds in drone imagery. The training dataset consisted of 294 images collected from Campbell River, Nanaimo, and Cowichan on Vancouver Island, Canada. After preprocessing, including augmentation and resizing, the dataset was expanded to 618 images to improve model performance. A user-friendly Streamlit application has been created, allowing researchers to upload drone images, visualize detections with annotated bounding boxes, and download per-image and per-location pinniped counts for further analysis.

![alt text](https://raw.githubusercontent.com/riyaeliza123/psf-image/refs/heads/main/2e26314d2df5fde18e421aa49392d6932f005eae4618f9d4149d15f7.jpg)

#### Test the model here: https://riyaeliza123-pinniped-drone-app.streamlit.app/

## Model metrics:
- Precision: 94.4%
- Recall: 72.7%
- Accuracy: 80.0%
- False positive rate: ~7.5%
- Area under ROC curve = 0.83,  indicates very good model performance (0.8 – 0.9 = strong discrimination ability)

Confusion matrix:
| | **Predicted Positive** | **Predicted Negative** | **Total Actual** |
|-------------------------|--------------|--------------|------------------|
| **Actual Positive**     | TP = 284     | FN = 107      | 391              |
| **Actual Negative**     | FP = 17      | TN = 210     | 227              |
| **Total Predicted**     | 301          | 317          | 618              |

## Modularized scripts and their functions:

| File             | Responsibility             | Key Functions                                         |
| ---------------- | -------------------------- | ----------------------------------------------------- |
| `config.py`      | Configuration & model init | Constants, model                                      |
| `exif_utils.py`  | EXIF & GPS extraction      | `extract_gps_from_image()`, `get_capture_date_time()` |
| `image_utils.py` | Image scaling & GSD        | `limit_resolution_to_temp()`, `compute_gsd()`         |
| `detection.py`   | Inference & parsing        | `run_detection()`, `parse_roboflow_detections()`      |
| `clustering.py`  | Unique count estimation    | `compute_unique_counts()`                             |
| `summaries.py`   | Output & CSV               | `display_and_download_summary()`                      |
| `app.py`         | Streamlit UI               | User interface + orchestration                        |

