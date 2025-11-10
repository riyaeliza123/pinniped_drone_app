# Handle resizing for upload constraints and calculate image scale (GSD)

import os, cv2, tempfile
from scripts.config import MAX_PIXELS, MAX_SIZE_MB, MIN_SCALE_PERCENT, CAMERA_SENSOR_WIDTHS
from scripts.exif_utils import get_float

def compute_gsd(tags, img_width):
    alt = get_float(tags.get("GPS GPSAltitude"))
    focal = get_float(tags.get("EXIF FocalLength"))
    if alt is None: alt = 20.0
    if focal is None: focal = 24.0
    model_name = str(tags.get("Image Model") or "Unknown").strip()
    sensor_w = CAMERA_SENSOR_WIDTHS.get(model_name, 13.2)
    if focal == 0 or img_width == 0:
        return float("inf")
    return (alt * sensor_w) / (focal * img_width)

def limit_resolution_to_temp(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    pixel_count = h * w

    if pixel_count <= MAX_PIXELS:
        return image_path, 1.0

    scale_percent = 80
    while scale_percent >= MIN_SCALE_PERCENT:
        width = int(w * scale_percent / 100)
        height = int(h * scale_percent / 100)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if resized.shape[0] * resized.shape[1] <= MAX_PIXELS:
            return temp_path, (w / width)

        os.remove(temp_path)
        scale_percent -= 10

    return temp_path, (w / width)

def progressive_resize_to_temp(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale_percent = 80
    while scale_percent >= MIN_SCALE_PERCENT:
        new_w = int(w * scale_percent / 100)
        new_h = int(h * scale_percent / 100)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        if size_mb <= MAX_SIZE_MB:
            return tmp.name, (w / new_w)
        os.remove(tmp.name)
        scale_percent -= 10
    return tmp.name, (w / new_w)
