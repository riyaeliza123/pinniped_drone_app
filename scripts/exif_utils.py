# Handle metadata extraction, GPS conversion, timestamps, and region labeling

import exifread
from datetime import datetime

def get_float(val):
    try:
        if hasattr(val, "values"):
            val = val.values[0]
        if hasattr(val, "num") and hasattr(val, "den"):
            return float(val.num) / float(val.den)
        if isinstance(val, str) and "/" in val:
            num, den = map(float, val.split("/"))
            return num / den
        return float(val)
    except:
        return None

def dms_to_decimal(dms, ref):
    degrees = get_float(dms[0])
    minutes = get_float(dms[1])
    seconds = get_float(dms[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal

def extract_gps_from_image(path):
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
        lat = dms_to_decimal(tags["GPS GPSLatitude"].values, tags["GPS GPSLatitudeRef"].values)
        lon = dms_to_decimal(tags["GPS GPSLongitude"].values, tags["GPS GPSLongitudeRef"].values)
        return lat, lon
    return None, None

def get_capture_date_time(path):
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if "Image DateTime" in tags:
        raw = str(tags["Image DateTime"])
        dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        return dt.date(), dt.time()
    return None, None

def get_location_name(lat):
    if 48 <= lat < 49:
        return "Cowichan"
    elif 49 <= lat < 50:
        return "Nanaimo"
    elif 50 <= lat < 51:
        return "Campbell River"
    else:
        return "Unknown"
