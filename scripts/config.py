# Centralize configuration constants and Roboflow model setup

import streamlit as st
from roboflow import Roboflow

# Roboflow model parameters
PROJECT_NAME = "pinnipeds-drone-imagery"
MODEL_VERSION = 18

# Image processing constraints
MAX_PIXELS = 4_000_000
MAX_SIZE_MB = 15
MIN_SCALE_PERCENT = 10

# Camera sensor widths for GSD calculations
CAMERA_SENSOR_WIDTHS = {
    "L2D-20c": 13.2,
    "FC3411": 13.2,
    "FC220": 6.3,
}

try:
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT_NAME)
    model = project.version(MODEL_VERSION).model
except Exception as e:
    st.error(f"Roboflow error: {e}")