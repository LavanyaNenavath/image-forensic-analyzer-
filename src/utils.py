"""
Image Forensic Analyzer - Utility Functions
Provides helper utilities for image I/O, format conversion,
image validation, forensic metric aggregation, and result visualization.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFile

# Allow loading slightly damaged/truncated images common from web scrapers and social media
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allowed forensic image formats
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"}
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


def validate_image(image_path_or_file):
    """
    Validates whether an input (file path, Streamlit UploadedFile, BytesIO, or bytes)
    is a valid, readable, uncorrupted image suitable for forensic analysis.

    Requirements enforced:
    1. Input must not be None or empty (0 bytes).
    2. Format must be one of JPG, JPEG, PNG, WEBP, BMP, or TIFF.
    3. File must decode without corruption or truncated streams.
    4. Image dimensions must meet minimum forensic requirements (at least 16x16 px).

    Returns:
        tuple: (is_valid: bool, error_message: str, pil_img: Image or None, np_img: np.ndarray or None)
    """
    if image_path_or_file is None:
        return False, "No image file provided. Please upload or select an image.", None, None

    # Check file path existence and size
    if isinstance(image_path_or_file, str):
        if not os.path.exists(image_path_or_file):
            return False, f"File does not exist: {image_path_or_file}", None, None
        try:
            file_size = os.path.getsize(image_path_or_file)
            if file_size == 0:
                return False, "Selected image file is empty (0 bytes).", None, None
        except Exception as e:
            return False, f"Cannot access file: {str(e)}", None, None

    # Check stream / UploadedFile size
    elif hasattr(image_path_or_file, "size"):
        if image_path_or_file.size == 0:
            return False, "Uploaded image file is empty (0 bytes).", None, None
    elif hasattr(image_path_or_file, "getvalue"):
        if len(image_path_or_file.getvalue()) == 0:
            return False, "Image buffer is empty (0 bytes).", None, None
    elif isinstance(image_path_or_file, (bytes, bytearray)):
        if len(image_path_or_file) == 0:
            return False, "Image byte data is empty (0 bytes).", None, None

    # Attempt to load and verify with load_image_rgb
    try:
        pil_img, np_img = load_image_rgb(image_path_or_file)
    except Exception as e:
        return False, f"Corrupted or invalid image: {str(e)}", None, None

    # Verify dimensions
    h, w = np_img.shape[:2]
    if w < 16 or h < 16:
        return (
            False,
            f"Image dimensions ({w}x{h} px) are too small. Forensic algorithms require at least 16x16 pixels.",
            None,
            None
        )

    # Verify channels and pixel type
    if np_img.ndim != 3 or np_img.shape[2] != 3:
        return False, "Image could not be normalized to a standard 3-channel RGB representation.", None, None

    return True, "", pil_img, np_img


def load_image_rgb(image_path_or_file):
    """
    Loads an image from a file path, Streamlit UploadedFile, BytesIO, bytes,
    or PIL Image, and returns:
    1. A PIL Image in RGB format
    2. A NumPy array in RGB format (uint8)

    Features:
    - Automatically handles EXIF orientation rotation.
    - Accurately converts transparency (RGBA, LA, P with alpha) using white background compositing.
    - Supports JPG, JPEG, PNG, WEBP, BMP, and TIFF formats.
    - Rewinds file pointers so caller buffers are never exhausted.
    """
    pil_img = None
    stream_to_rewind = None

    if isinstance(image_path_or_file, str):
        if not os.path.exists(image_path_or_file):
            raise FileNotFoundError(f"Image not found at: {image_path_or_file}")
        pil_img = Image.open(image_path_or_file)

    elif isinstance(image_path_or_file, Image.Image):
        pil_img = image_path_or_file.copy()

    elif isinstance(image_path_or_file, np.ndarray):
        if image_path_or_file.ndim == 2:
            pil_img = Image.fromarray(image_path_or_file).convert("RGB")
        elif image_path_or_file.ndim == 3:
            if image_path_or_file.shape[2] == 4:
                pil_img = Image.fromarray(image_path_or_file, mode="RGBA")
            else:
                pil_img = Image.fromarray(image_path_or_file, mode="RGB")
        else:
            raise ValueError(f"Unsupported array shape: {image_path_or_file.shape}")

    elif isinstance(image_path_or_file, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_path_or_file))

    elif hasattr(image_path_or_file, "read"):
        # Streamlit UploadedFile or io.BytesIO
        stream_to_rewind = image_path_or_file
        if hasattr(stream_to_rewind, "seek"):
            stream_to_rewind.seek(0)
        # Read into BytesIO to avoid underlying socket/stream lock issues
        data = stream_to_rewind.read()
        pil_img = Image.open(io.BytesIO(data))
        if hasattr(stream_to_rewind, "seek"):
            stream_to_rewind.seek(0)

    else:
        raise TypeError(f"Unsupported image input type: {type(image_path_or_file)}")

    # Force decoding of image pixel buffer into RAM to catch file truncation errors early
    pil_img.load()

    # Handle EXIF auto-rotation
    try:
        transposed = ImageOps.exif_transpose(pil_img)
        if transposed is not None:
            pil_img = transposed
    except Exception:
        pass

    # Handle Alpha channel & Color Modes cleanly
    # For transparent images (PNG, WebP with alpha), composite over white background
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba_img = pil_img.convert("RGBA")
        background = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        alpha_composite = Image.alpha_composite(background, rgba_img)
        pil_img = alpha_composite.convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Generate standard NumPy RGB array
    np_img = np.array(pil_img, dtype=np.uint8)

    # Guarantee (H, W, 3)
    if np_img.ndim == 2:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    elif np_img.ndim == 3 and np_img.shape[2] == 4:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)

    return pil_img, np_img


def pil_to_cv2(pil_img):
    """Converts a PIL RGB Image to an OpenCV BGR NumPy array."""
    rgb_arr = np.array(pil_img)
    if rgb_arr.ndim == 2:
        return cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
    elif rgb_arr.shape[2] == 4:
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_bgr):
    """Converts an OpenCV BGR NumPy array to a PIL RGB Image."""
    if cv2_bgr.ndim == 2:
        rgb_arr = cv2.cvtColor(cv2_bgr, cv2.COLOR_GRAY2RGB)
    elif cv2_bgr.shape[2] == 4:
        rgb_arr = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb_arr = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_arr)


def compute_image_statistics(np_rgb):
    """
    Computes statistical properties of an image:
    mean intensity, standard deviation, sharpness, and entropy.
    Safely converts all metrics to standard Python native types.
    """
    if np_rgb.ndim == 3 and np_rgb.shape[2] == 3:
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    elif np_rgb.ndim == 3 and np_rgb.shape[2] == 4:
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGBA2GRAY)
    elif np_rgb.ndim == 2:
        gray = np_rgb
    else:
        gray = np_rgb[:, :, 0]

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Laplacian sharpness variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var()) if lap.size > 0 else 0.0

    # Shannon Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

    return {
        "mean_brightness": round(mean_val, 2),
        "std_deviation": round(std_val, 2),
        "sharpness_var": round(sharpness, 2),
        "entropy": round(entropy, 3),
        "height": int(np_rgb.shape[0]),
        "width": int(np_rgb.shape[1]),
        "channels": int(np_rgb.shape[2]) if np_rgb.ndim == 3 else 1
    }


def create_heatmap_overlay(original_rgb, difference_gray, alpha=0.55, colormap=cv2.COLORMAP_JET):
    """
    Generates a colorized heatmap overlay showing high difference regions
    blended on top of the original image.
    """
    if difference_gray.dtype != np.uint8:
        norm_diff = cv2.normalize(difference_gray, None, 0, 255, cv2.NORM_MINMAX)
        norm_diff = norm_diff.astype(np.uint8)
    else:
        norm_diff = difference_gray

    colored_heatmap_bgr = cv2.applyColorMap(norm_diff, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    if colored_heatmap_rgb.shape[:2] != original_rgb.shape[:2]:
        colored_heatmap_rgb = cv2.resize(
            colored_heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0])
        )

    orig_clean = original_rgb
    if orig_clean.ndim == 2:
        orig_clean = cv2.cvtColor(orig_clean, cv2.COLOR_GRAY2RGB)
    elif orig_clean.shape[2] == 4:
        orig_clean = cv2.cvtColor(orig_clean, cv2.COLOR_RGBA2RGB)

    blended = cv2.addWeighted(colored_heatmap_rgb, alpha, orig_clean, 1.0 - alpha, 0)
    return blended


def calculate_overall_verdict(ela_score, copy_move_matches, noise_score, exif_suspicious):
    """
    Multi-Modal Forensic Decision Fusion:
    In digital forensics, different tampering attacks (copy-move, splicing, retouching)
    trigger different specialized indicators:

    1. Copy-Move: Triggers keypoint duplicate matches and spatial clusters.
    2. Splicing / Noise: Triggers high-frequency noise variance discrepancy across blocks.
    3. Compression/Retouching: Triggers ELA anomaly and compression discrepancy patterns.
    4. EXIF: Detects editing software footprints (Photoshop/GIMP). EXIF metadata serves
       as supporting evidence, not the primary score.

    The ensemble evaluates the maximum indicator score and adds corroboration bonuses
    when multiple forensic techniques identify concurrent anomalies.
    The resulting score is a rule-based forensic risk/suspicion score, not a calibrated
    statistical probability or confidence ratio.
    """
    # 1. Copy-Move normalized score (0 - 100)
    if copy_move_matches >= 8:
        cm_score = min(98.0, 75.0 + (copy_move_matches * 1.5))
    elif copy_move_matches >= 3:
        cm_score = 55.0 + (copy_move_matches * 5.0)
    elif copy_move_matches >= 1:
        cm_score = 25.0
    else:
        cm_score = 5.0

    # 2. Splicing / Noise normalized score (0 - 100)
    # The noise_score parameter is the noise inconsistency score (0 - 100),
    # where an authentic baseline image typically exhibits low inconsistency (< 35.0).
    if noise_score >= 50.0:
        splice_score = min(98.0, float(noise_score))
    elif noise_score >= 35.0:
        splice_score = 50.0 + (float(noise_score) - 35.0) * 1.5
    else:
        splice_score = (float(noise_score) / 35.0) * 20.0

    # 3. ELA normalized score (0 - 100)
    # Normal authentic JPEG ELA score is typically < 10.0
    if ela_score >= 25.0:
        ela_norm = min(95.0, 60.0 + (float(ela_score) - 25.0) * 1.5)
    elif ela_score >= 12.0:
        ela_norm = 40.0 + (float(ela_score) - 12.0) * 1.5
    else:
        ela_norm = (float(ela_score) / 12.0) * 20.0

    # 4. EXIF bonus (EXIF metadata provides supporting evidence, not primary score)
    exif_score = 75.0 if exif_suspicious else 0.0

    # Primary indicator dominance
    primary_scores = [cm_score, splice_score, ela_norm]
    max_primary = max(primary_scores)

    # Corroboration: count how many forensic tests detect anomalies
    anomalous_tests = sum(1 for s in primary_scores if s >= 50.0)
    if exif_suspicious:
        anomalous_tests += 1

    corroboration_bonus = max(0.0, (anomalous_tests - 1) * 8.0)
    overall_confidence = min(99.0, max_primary + corroboration_bonus)

    # Classification logic
    if overall_confidence >= 65.0:
        verdict = "HIGH FORENSIC RISK / POTENTIAL TAMPERING"
        risk_level = "HIGH"
        status_color = "#4F46E5"  # Electric Indigo (non-red alert)
        if cm_score >= 60.0 and cm_score == max_primary:
            primary_technique = "Copy-Move Cloned Duplication"
        elif splice_score >= 50.0 and splice_score == max_primary:
            primary_technique = "Potential Noise Inconsistency / Splicing"
        else:
            primary_technique = "Compression / Error Level Anomaly"
    elif overall_confidence >= 40.0:
        verdict = "SUSPICIOUS / POSSIBLE EDITING"
        risk_level = "MODERATE"
        status_color = "#FB8C00"  # Orange
        primary_technique = "Subtle Discrepancy Detected"
    else:
        verdict = "NO SIGNIFICANT FORENSIC INDICATORS"
        risk_level = "LOW"
        status_color = "#43A047"  # Green
        primary_technique = "Uniform Baseline (Consistent)"

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "forensic_risk_score": round(float(overall_confidence), 1),
        "status_color": status_color,
        "primary_technique": primary_technique,
        "breakdown": {
            "ela_score": round(float(ela_score), 2),
            "copy_move_matches": int(copy_move_matches),
            "noise_inconsistency": round(float(noise_score), 2),
            "exif_suspicious": bool(exif_suspicious)
        }
    }
