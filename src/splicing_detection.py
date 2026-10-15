"""
Image Forensic Analyzer - Image Splicing & Noise Inconsistency Detection
------------------------------------------------------------------------
Theoretical Foundation:
Image Splicing involves inserting a patch from an external donor image into a recipient
host image. Every camera sensor imparts distinct Photo-Response Non-Uniformity (PRNU)
and ISO noise distribution.

Forensic Methodology:
1. High-Frequency Noise Extraction:
   Applies a non-linear median filter and computes the high-frequency residual:
   Residual = |Image - MedianFilter(Image)|
2. Local Noise Variance Mapping:
   Divides the residual into blocks (e.g. 16x16 pixels) and measures the local variance.
3. Robust Statistical Anomaly Scoring:
   In authentic images, natural sensor noise exhibits a uniform distribution where
   the ratio between the 95th percentile variance and median variance (P95 / Median)
   is modest (typically <= 3.0).
   When an external element is spliced in, foreign noise variance causes an extreme
   outlier spike in this ratio (> 5.0 to 30.0+), exposing the tampering.
"""

import cv2
import numpy as np


def detect_splicing_noise(image_rgb, kernel_size=3, block_size=16):
    """
    Analyzes noise inconsistency across image blocks to detect spliced patches
    and foreign object insertions.

    Args:
        image_rgb: NumPy array (H, W, 3) in RGB, (H, W) grayscale, or (H, W, 4) RGBA.
        kernel_size (int): Size of median filter kernel (odd positive integer).
        block_size (int): Dimension of local variance estimation blocks.

    Returns:
        dict containing:
            - 'noise_residual_rgb': Enhanced high-frequency noise residual map
            - 'variance_heatmap_rgb': Colorized heatmap showing noise distribution
            - 'laplacian_residual_rgb': Laplacian edge high-pass map
            - 'noise_inconsistency_score': Quantitative score (0-100)
            - 'is_splicing_detected': Boolean verdict
            - 'variance_ratio': Ratio of P95 to median variance
            - 'mean_noise_level': Average noise level
            - 'noise_std': Standard deviation of noise
    """
    # Step 0: Channel normalization
    if image_rgb.ndim == 3:
        if image_rgb.shape[2] == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        elif image_rgb.shape[2] == 4:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2GRAY).astype(np.float32)
        else:
            gray = image_rgb[:, :, 0].astype(np.float32)
    elif image_rgb.ndim == 2:
        gray = image_rgb.astype(np.float32)
    else:
        raise ValueError(f"Invalid image array dimension: {image_rgb.ndim}")

    h, w = gray.shape[:2]

    # Guard for odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, min(h, w) - (1 if min(h, w) % 2 == 0 else 0)))

    # Step 1: Extract high-frequency noise residual via median subtraction
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    median_filtered = cv2.medianBlur(gray_u8, kernel_size).astype(np.float32)
    noise_residual = np.abs(gray - median_filtered)

    # Step 2: Adaptive block-wise local variance analysis
    effective_block_size = max(2, min(block_size, h // 2, w // 2))
    rows = h // effective_block_size
    cols = w // effective_block_size

    # Handle extremely small images where rows or cols would be 0
    if rows < 1 or cols < 1:
        global_var = float(np.var(noise_residual))
        return {
            "noise_residual_rgb": cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB),
            "variance_heatmap_rgb": cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB),
            "laplacian_residual_rgb": cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB),
            "noise_inconsistency_score": 10.0,
            "is_splicing_detected": False,
            "variance_ratio": 1.0,
            "mean_noise_level": round(float(np.mean(noise_residual)), 2),
            "noise_std": round(float(np.std(noise_residual)), 2)
        }

    # Vectorized fast block variance computation
    cropped = noise_residual[:rows * effective_block_size, :cols * effective_block_size]
    blocks = cropped.reshape(rows, effective_block_size, cols, effective_block_size).swapaxes(1, 2).reshape(rows, cols, -1)
    variance_map = np.var(blocks, axis=-1).astype(np.float32)

    # Step 3: Statistical Discrepancy (P95 to Median Variance Ratio)
    flat_vars = variance_map.flatten()
    if len(flat_vars) == 0:
        med_var = 0.0
        p95_var = 0.0
        var_ratio = 1.0
    else:
        med_var = float(np.median(flat_vars))
        p95_var = float(np.percentile(flat_vars, 95))
        var_ratio = float(p95_var / (med_var + 1e-4))

    # Scoring formulation:
    # Authentic baseline: ratio <= 2.5 -> score < 20
    # Moderate anomaly: ratio 3.0 to 5.0 -> score 35 to 60
    # Heavy splicing: ratio >= 6.0 -> score 75 to 99
    if var_ratio >= 6.0:
        score = min(99.0, 75.0 + (var_ratio - 6.0) * 0.8)
    elif var_ratio >= 3.0:
        score = 40.0 + (var_ratio - 3.0) * 11.5
    else:
        score = (var_ratio / 3.0) * 20.0

    inconsistency_score = round(float(min(99.0, max(1.0, score))), 1)
    is_splicing_detected = bool(inconsistency_score >= 50.0)

    # Step 4: Visualizations
    if np.max(variance_map) > np.min(variance_map):
        norm_variance = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        norm_variance = np.zeros_like(variance_map, dtype=np.uint8)

    if norm_variance.size > 0:
        upscaled_variance = cv2.resize(norm_variance, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled_variance = np.zeros((h, w), dtype=np.uint8)

    variance_heatmap_bgr = cv2.applyColorMap(upscaled_variance, cv2.COLORMAP_INFERNO)
    variance_heatmap_rgb = cv2.cvtColor(variance_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Laplacian edge residual
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_abs = np.abs(laplacian)
    if np.max(laplacian_abs) > np.min(laplacian_abs):
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        laplacian_norm = np.zeros((h, w), dtype=np.uint8)
    laplacian_rgb = cv2.cvtColor(laplacian_norm, cv2.COLOR_GRAY2RGB)

    # Enhanced residual
    if np.max(noise_residual) > np.min(noise_residual):
        enhanced_residual = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        enhanced_residual = np.zeros((h, w), dtype=np.uint8)
    enhanced_residual_rgb = cv2.cvtColor(enhanced_residual, cv2.COLOR_GRAY2RGB)

    return {
        "noise_residual_rgb": enhanced_residual_rgb,
        "variance_heatmap_rgb": variance_heatmap_rgb,
        "laplacian_residual_rgb": laplacian_rgb,
        "noise_inconsistency_score": inconsistency_score,
        "is_splicing_detected": is_splicing_detected,
        "variance_ratio": round(float(var_ratio), 2),
        "mean_noise_level": round(float(np.mean(noise_residual)), 2),
        "noise_std": round(float(np.std(noise_residual)), 2)
    }
