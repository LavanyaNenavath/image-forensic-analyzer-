"""
Image Forensic Analyzer - EXIF Metadata & Digital Header Forensics
-------------------------------------------------------------------
Theoretical Foundation:
Digital cameras write rich Exchangeable Image File Format (EXIF) metadata into
the image headers, including camera hardware specifications (Make, Model, Lens),
exposure parameters (ISO, Aperture, Shutter Speed), and timestamps.

When an image is modified in photo-manipulation software (e.g., Photoshop, GIMP):
1. The software often injects its name into tags like 'Software', 'ProcessingSoftware',
   or Adobe XMP metadata.
2. Timestamps often become desynchronized: DateTimeOriginal != DateTime (modification).
3. Critical camera hardware profiles may be stripped or replaced by generic values.
4. PNG, WebP, and TIFF files may encode editing signatures in header text chunks.
"""

import os
import io
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
try:
    from PIL.ExifTags import IFD
except ImportError:
    IFD = None


# Known digital editing software signatures
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "paint.net", "canva",
    "picsart", "snapseed", "pixlr", "affinity", "corel",
    "seashore", "vsco", "facetune", "retouch", "editor",
    "photopea", "procreate", "photofox", "fotor", "befunky",
    "pixelmator", "krita", "polarr", "vanceai"
]


def _safe_str(val):
    """Converts EXIF values (bytes, tuples, objects) into clean, human-readable strings."""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip("\x00 \t\r\n")
        except Exception:
            return f"<Binary Data {len(val)} bytes>"
    if isinstance(val, (tuple, list)):
        try:
            return ", ".join(str(_safe_str(x)) for x in val)
        except Exception:
            return str(val)
    return str(val).strip()


def extract_metadata(image_input):
    """
    Extracts and parses EXIF metadata and header metadata from an image,
    performing forensic auditing for editing software footprints, timestamp anomalies,
    and digital provenance.

    Supports:
        - File path (str)
        - Streamlit UploadedFile
        - io.BytesIO or file-like buffer
        - Raw bytes or bytearray
        - PIL.Image.Image object
        - NumPy array (fallback)

    Returns:
        dict containing:
            - 'exif_found': bool
            - 'metadata_found': bool
            - 'tags': dict of human-readable tags
            - 'is_suspicious': bool
            - 'reasons': list of identified anomalies
            - 'software_detected': str or None
            - 'camera_model': str or None
            - 'timestamp_original': str or None
            - 'timestamp_modified': str or None
            - 'file_format': str
            - 'image_size': str
            - 'color_mode': str
    """
    pil_img = None
    file_format = "Unknown"
    image_size = "Unknown"
    color_mode = "Unknown"
    close_needed = False

    # 1. Resolve image_input to a PIL Image safely
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {
                    "exif_found": False,
                    "metadata_found": False,
                    "is_suspicious": False,
                    "reasons": ["File not found at specified path."],
                    "tags": {},
                    "software_detected": None,
                    "camera_model": None,
                    "timestamp_original": None,
                    "timestamp_modified": None,
                    "file_format": "None",
                    "image_size": "0x0",
                    "color_mode": "None"
                }
            pil_img = Image.open(image_input)
            close_needed = True

        elif isinstance(image_input, Image.Image):
            pil_img = image_input

        elif isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input))
            close_needed = True

        elif hasattr(image_input, "read"):
            # Streamlit UploadedFile, io.BytesIO, or file stream
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            pil_img = Image.open(image_input)
            close_needed = True
            # Rewind so caller stream position is preserved
            if hasattr(image_input, "seek"):
                image_input.seek(0)

        elif hasattr(image_input, "shape"):
            # NumPy array (contains no EXIF headers)
            h, w = image_input.shape[:2]
            return {
                "exif_found": False,
                "metadata_found": False,
                "is_suspicious": False,
                "reasons": ["Raw in-memory image array (no EXIF header present)."],
                "tags": {"Dimensions": f"{w}x{h}", "Type": "NumPy Array"},
                "software_detected": None,
                "camera_model": None,
                "timestamp_original": None,
                "timestamp_modified": None,
                "file_format": "Raw Array",
                "image_size": f"{w}x{h}",
                "color_mode": "RGB" if (len(image_input.shape) == 3 and image_input.shape[2] == 3) else "Grayscale"
            }
        else:
            return {
                "exif_found": False,
                "metadata_found": False,
                "is_suspicious": False,
                "reasons": ["Unsupported image input type."],
                "tags": {},
                "software_detected": None,
                "camera_model": None,
                "timestamp_original": None,
                "timestamp_modified": None,
                "file_format": "Unknown",
                "image_size": "Unknown",
                "color_mode": "Unknown"
            }
    except Exception as e:
        return {
            "exif_found": False,
            "metadata_found": False,
            "is_suspicious": False,
            "reasons": [f"Failed to read image headers: {str(e)}"],
            "tags": {},
            "software_detected": None,
            "camera_model": None,
            "timestamp_original": None,
            "timestamp_modified": None,
            "file_format": "Error",
            "image_size": "Error",
            "color_mode": "Error"
        }

    # Extract basic format properties
    try:
        file_format = getattr(pil_img, "format", None) or "Unknown"
        color_mode = getattr(pil_img, "mode", None) or "Unknown"
        if hasattr(pil_img, "width") and hasattr(pil_img, "height"):
            image_size = f"{pil_img.width}x{pil_img.height}"
    except Exception:
        pass

    parsed_tags = {}
    exif_found = False

    # 2. Extract standard EXIF tags via getexif()
    try:
        if hasattr(pil_img, "getexif"):
            exif_obj = pil_img.getexif()
            if exif_obj and len(exif_obj) > 0:
                exif_found = True
                for tag_id, val in exif_obj.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    parsed_tags[tag_name] = _safe_str(val)

                # Check Sub-IFDs if available (ExifIFD, GPSInfo)
                if IFD is not None:
                    # Sub-IFD: Exif (where DateTimeOriginal, LensModel, etc. live)
                    try:
                        exif_sub = exif_obj.get_ifd(IFD.Exif)
                        for sub_id, sub_val in exif_sub.items():
                            sub_name = TAGS.get(sub_id, str(sub_id))
                            if sub_name not in parsed_tags:
                                parsed_tags[sub_name] = _safe_str(sub_val)
                    except Exception:
                        pass

                    # Sub-IFD: GPSInfo
                    try:
                        gps_sub = exif_obj.get_ifd(IFD.GPSInfo)
                        for gps_id, gps_val in gps_sub.items():
                            gps_name = GPSTAGS.get(gps_id, str(gps_id))
                            parsed_tags[f"GPS_{gps_name}"] = _safe_str(gps_val)
                    except Exception:
                        pass
    except Exception:
        pass

    # 3. Fallback for legacy JPEG _getexif() if getexif() was empty
    if not exif_found and hasattr(pil_img, "_getexif"):
        try:
            legacy_data = pil_img._getexif()
            if legacy_data:
                exif_found = True
                for tag_id, val in legacy_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if tag_name not in parsed_tags:
                        parsed_tags[tag_name] = _safe_str(val)
        except Exception:
            pass

    # 4. Extract format-specific metadata from pil_img.info (PNG tEXt, WebP, TIFF, XMP)
    metadata_found = exif_found
    try:
        if hasattr(pil_img, "info") and isinstance(pil_img.info, dict):
            for k, v in pil_img.info.items():
                if k in ("exif", "icc_profile"):
                    continue  # Raw binary dumps
                k_str = str(k)
                if k_str not in parsed_tags:
                    cleaned_val = _safe_str(v)
                    # Skip excessively huge strings (e.g. embedded thumbnails/XML dumps > 2000 chars)
                    if len(cleaned_val) > 2000:
                        cleaned_val = cleaned_val[:2000] + "... [truncated]"
                    parsed_tags[k_str] = cleaned_val
                    metadata_found = True
    except Exception:
        pass

    # 5. Forensic Auditing for Software Signatures & Timestamps
    reasons = []
    is_suspicious = False
    software_detected = None

    # Inspect all parsed tags and info for editing software signatures
    for tag_name, tag_val in parsed_tags.items():
        tag_val_lower = str(tag_val).lower()
        # Look specifically in Software, ProcessingSoftware, Comment, ImageDescription, or any tag
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in tag_val_lower:
                is_suspicious = True
                if not software_detected:
                    software_detected = str(tag_val)
                reason_str = f"Image header ({tag_name}) matches known editor: '{tag_val}'"
                if reason_str not in reasons:
                    reasons.append(reason_str)
                break

    # Check for Camera Make/Model
    camera_make = parsed_tags.get("Make")
    camera_model = parsed_tags.get("Model")
    if camera_make or camera_model:
        cam_info = f"{camera_make or ''} {camera_model or ''}".strip()
    else:
        cam_info = None

    # Check for Timestamp Desynchronization
    time_orig = parsed_tags.get("DateTimeOriginal")
    time_mod = parsed_tags.get("DateTime")
    if time_orig and time_mod and (time_orig != time_mod):
        is_suspicious = True
        reasons.append(f"Timestamp Mismatch: Original ({time_orig}) != Modified ({time_mod})")

    # If no overt editing detected, note clean state
    if not reasons:
        if metadata_found:
            reasons = ["Headers present with no overt editing software signatures."]
        else:
            reasons = ["No EXIF metadata present (standard for screenshots and social media platforms like WhatsApp, X/Twitter, Instagram)."]

    return {
        "exif_found": bool(exif_found),
        "metadata_found": bool(metadata_found),
        "is_suspicious": bool(is_suspicious),
        "reasons": reasons,
        "tags": parsed_tags,
        "software_detected": software_detected,
        "camera_model": cam_info,
        "timestamp_original": time_orig,
        "timestamp_modified": time_mod,
        "file_format": str(file_format),
        "image_size": str(image_size),
        "color_mode": str(color_mode)
    }
