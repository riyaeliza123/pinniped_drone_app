# Pinniped census app
Computer vision project to count number of pinnipeds in each drone image.
294 images used for training from Campbell, Nanimo and Cowichan (Vancouver Island, Canada). Total 618 images after pre-proessing for training. 
Test the model here: https://riyaeliza123-pinniped-drone-app.streamlit.app/

Model metrics:
- Precision: 88.6%
- Recall: 80.1%
- Accuracy: 82.9%
- False positive rate: ~11 %
- Area under ROC curve = 0.83,  indicates very good model performance (0.8 – 0.9 = strong discrimination ability)

Confusion matrix:
| | **Predicted Positive** | **Predicted Negative** | **Total Actual** |
|-------------------------|--------------|--------------|------------------|
| **Actual Positive**     | TP = 284     | FN = 71      | 355              |
| **Actual Negative**     | FP = 37      | TN = 227     | 264              |
| **Total Predicted**     | 321          | 298          | 618              |

| File             | Responsibility             | Key Functions                                         |
| ---------------- | -------------------------- | ----------------------------------------------------- |
| `config.py`      | Configuration & model init | Constants, model                                      |
| `exif_utils.py`  | EXIF & GPS extraction      | `extract_gps_from_image()`, `get_capture_date_time()` |
| `image_utils.py` | Image scaling & GSD        | `limit_resolution_to_temp()`, `compute_gsd()`         |
| `detection.py`   | Inference & parsing        | `run_detection()`, `parse_roboflow_detections()`      |
| `clustering.py`  | Unique count estimation    | `compute_unique_counts()`                             |
| `summaries.py`   | Output & CSV               | `display_and_download_summary()`                      |
| `app.py`         | Streamlit UI               | User interface + orchestration                        |

