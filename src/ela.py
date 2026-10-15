"""
Image Forensic Analyzer - Error Level Analysis (ELA) Module
------------------------------------------------------------
Theoretical Foundation:
JPEG compression divides an image into 8x8 pixel blocks and applies the Discrete
Cosine Transform (DCT) followed by quantization. Each re-compression at a specific
quality factor causes loss of high-frequency components until an error equilibrium
is achieved across the entire image.

If an image is modified (e.g. an object is pasted from another source, or a portion
is digitally retouched):
1. The untouched background will show lower error because it has already reached
   compression equilibrium.
2. The newly introduced or modified areas will show significantly higher or lower
   error levels because they have not undergone identical quantization cycles.

ELA works by:
1. Re-saving the input image at a known JPEG quality (typically 90% or 95%).
2. Computing the absolute pixel-wise difference: |Original - Resaved|.
3. Amplifying the difference using an enhancement scale factor (e.g., 10x-20x)
   so subtle discrepancies are visible to human and algorithmic inspectors.
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def perform_ela(image_input, quality=90, scale_factor=15):
    """
    Performs Error Level Analysis on the provided image.

    Args:
        image_input: File path (str), PIL.Image, NumPy array (RGB/RGBA/Grayscale),
                     or file-like buffer (UploadedFile, BytesIO).
        quality (int): Re-compression JPEG quality (typically 90-95).
        scale_factor (int or str): Multiplier to amplify subtle error differences.

    Returns:
        dict: A comprehensive dictionary containing:
            - 'ela_image_pil': Enhanced ELA PIL Image
            - 'ela_image_np': Enhanced ELA NumPy array (RGB)
            - 'diff_gray': 2D grayscale array of raw difference
            - 'heatmap_rgb': Colorized heatmap highlighting suspicious zones
            - 'mean_error': Average error per pixel
            - 'max_error': Maximum pixel error
            - 'std_error': Standard deviation of the error distribution
            - 'high_error_ratio': Percentage of pixels exhibiting high anomaly
            - 'ela_tampering_score': Normalized tampering index (0 to 100)
    """
    stream_to_rewind = None

    # Step 1: Normalize input to a clean PIL Image in RGB format
    if isinstance(image_input, str):
        original_pil = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            original_pil = Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:
            original_pil = Image.fromarray(image_input, mode="RGBA")
        else:
            original_pil = Image.fromarray(image_input, mode="RGB")
    elif isinstance(image_input, Image.Image):
        original_pil = image_input.copy()
    elif isinstance(image_input, (bytes, bytearray)):
        original_pil = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        stream_to_rewind = image_input
        if hasattr(stream_to_rewind, "seek"):
            stream_to_rewind.seek(0)
        data = stream_to_rewind.read()
        original_pil = Image.open(io.BytesIO(data))
        if hasattr(stream_to_rewind, "seek"):
            stream_to_rewind.seek(0)
    else:
        raise TypeError(f"Unsupported image input type for ELA: {type(image_input)}")

    # Handle transparent images cleanly before JPEG compression (which does not support alpha)
    if original_pil.mode in ("RGBA", "LA") or (original_pil.mode == "P" and "transparency" in original_pil.info):
        rgba_img = original_pil.convert("RGBA")
        background = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        original_pil = Image.alpha_composite(background, rgba_img).convert("RGB")
    elif original_pil.mode != "RGB":
        original_pil = original_pil.convert("RGB")

    # Step 2: Re-save image in memory at the specified JPEG quality factor
    buffer = io.BytesIO()
    original_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved_pil = Image.open(buffer).convert("RGB")

    # Step 3: Compute absolute difference between original and resaved image
    diff_pil = ImageChops.difference(original_pil, resaved_pil)

    # Step 4: Calculate extrema and dynamic error metrics
    extrema = diff_pil.getextrema()
    # Find the maximum difference across all three RGB channels
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1  # Guard against division by zero

    # Dynamic scaling factor calculation:
    if scale_factor is None or scale_factor == "auto":
        scale = int(255.0 / max_diff)
    else:
        scale = int(scale_factor)

    # Step 5: Amplify the difference image for human visualization
    enhanced_pil = ImageEnhance.Brightness(diff_pil).enhance(scale)
    ela_np = np.array(enhanced_pil, dtype=np.uint8)

    # Step 6: Convert raw difference to grayscale for quantitative analysis
    diff_np = np.array(diff_pil, dtype=np.float32)
    # Euclidean norm across RGB channels to get true intensity difference
    diff_gray = np.sqrt(np.mean(diff_np ** 2, axis=2)).astype(np.uint8)

    # Step 7: Statistical Feature Extraction
    mean_err = float(np.mean(diff_gray))
    std_err = float(np.std(diff_gray))
    max_err = float(np.max(diff_gray))

    # Anomaly threshold: pixels that deviate significantly above average noise
    anomaly_threshold = mean_err + 2.0 * std_err
    high_error_pixels = int(np.sum(diff_gray > max(15, anomaly_threshold)))
    total_pixels = max(1, diff_gray.shape[0] * diff_gray.shape[1])
    high_error_ratio = round((high_error_pixels / total_pixels) * 100.0, 3)

    # ELA Tampering Score Formulation:
    # In authentic photos, standard deviation is low and uniform.
    # In spliced/tampered photos, the variance and high-error ratio spike.
    score_metric = (mean_err * 2.0) + (std_err * 3.0) + (high_error_ratio * 4.0)
    ela_tampering_score = round(float(min(100.0, max(0.0, score_metric))), 2)

    # Step 8: Generate Colorized Heatmap (JET colormap)
    if np.max(diff_gray) > np.min(diff_gray):
        norm_diff = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        norm_diff = np.zeros_like(diff_gray, dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return {
        "ela_image_pil": enhanced_pil,
        "ela_image_np": ela_np,
        "diff_gray": diff_gray,
        "heatmap_rgb": heatmap_rgb,
        "mean_error": round(float(mean_err), 2),
        "max_error": round(float(max_err), 2),
        "std_error": round(float(std_err), 2),
        "high_error_ratio": float(high_error_ratio),
        "ela_tampering_score": float(ela_tampering_score),
        "quality_used": int(quality),
        "scale_factor_used": int(scale)
    }


def analyze_ela_regions(diff_gray, grid_size=(8, 8)):
    """
    Divides the error difference map into grid blocks and computes
    regional variances to identify localized tampering clusters.

    Args:
        diff_gray: 2D NumPy array of ELA differences
        grid_size: Tuple (rows, cols)

    Returns:
        tuple: (variance_matrix, anomalous_blocks_indices)
    """
    h, w = diff_gray.shape[:2]
    rows, cols = grid_size
    block_h = max(1, h // rows)
    block_w = max(1, w // cols)

    variance_matrix = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            block = diff_gray[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
            if block.size > 0:
                variance_matrix[r, c] = float(np.var(block))

    # Identify blocks whose variance is 1.8 standard deviations above the grid mean
    grid_mean = float(np.mean(variance_matrix))
    grid_std = float(np.std(variance_matrix))
    threshold = grid_mean + (1.8 * grid_std)

    anomalous_blocks = []
    if grid_std > 1e-4:
        for r in range(rows):
            for c in range(cols):
                if variance_matrix[r, c] > threshold:
                    anomalous_blocks.append((r, c, float(variance_matrix[r, c])))

    return variance_matrix, anomalous_blocks
