"""
Image Forensic Analyzer - Copy-Move Forgery Detection (ORB Feature Matching)
----------------------------------------------------------------------------
Theoretical Foundation:
Copy-Move (Cloning) Forgery occurs when a region from an image is duplicated
and pasted into another location within the same image.

Algorithm Overview:
1. ORB Keypoints & Descriptors:
   Detects distinctive corner and textural keypoints and generates 256-bit binary descriptors.
2. Self-Matching with k-NN (k=4):
   In self-matching, neighbor 0 is always the point matching itself (dist=0).
   By extracting k=4 nearest neighbors and discarding queryIdx == trainIdx,
   we obtain true non-self candidates:
   m1 = closest non-self match
   m2 = second closest non-self match
3. Spatial Distance Filter:
   Ensures matched points are separated by at least `min_distance` pixels
   to avoid trivial matches in uniform textures.
4. Lowe's Ratio Test:
   Accepts candidate pairs if m1.distance < ratio_threshold * m2.distance.
5. Spatial Displacement Clustering:
   Genuine duplicated regions share consistent displacement translation vectors (dx, dy).
"""

import cv2
import numpy as np


def detect_copy_move(image_rgb, max_features=3000, min_distance=35, ratio_threshold=0.82):
    """
    Detects copy-move forgery using ORB feature matching with self-match rejection
    and translation displacement clustering.

    Args:
        image_rgb: NumPy array (H, W, 3) in RGB, (H, W) grayscale, or (H, W, 4) RGBA.
        max_features (int): Number of ORB keypoints.
        min_distance (int): Minimum Euclidean pixel distance between matched points.
        ratio_threshold (float): Lowe's ratio test threshold.

    Returns:
        dict:
            - 'annotated_image': RGB canvas with plotted duplicate pairs & connection lines
            - 'match_count': Number of verified duplicated keypoint pairs
            - 'is_tampered': Boolean verdict
            - 'confidence': Confidence score (0 to 100%)
            - 'clusters': Count of spatial displacement clusters
            - 'keypoints_found': Total keypoints detected
    """
    # Channel normalization
    if image_rgb.ndim == 3:
        if image_rgb.shape[2] == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            annotated = image_rgb.copy()
        elif image_rgb.shape[2] == 4:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2GRAY)
            annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        else:
            gray = image_rgb[:, :, 0]
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif image_rgb.ndim == 2:
        gray = image_rgb.copy()
        annotated = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Invalid image array dimension: {image_rgb.ndim}")

    # Step 1: Initialize ORB detector
    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=18
    )

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 8:
        return {
            "annotated_image": annotated,
            "match_count": 0,
            "is_tampered": False,
            "confidence": 0.0,
            "clusters": 0,
            "keypoints_found": len(keypoints) if keypoints else 0
        }

    # Step 2: k-NN matching with k=4 (to account for self-matching)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors, descriptors, k=min(4, len(descriptors)))

    valid_pairs = []
    seen_pairs = set()

    for match_list in raw_matches:
        # Filter out self matches (queryIdx == trainIdx)
        non_self = [m for m in match_list if m.queryIdx != m.trainIdx]
        if len(non_self) < 2:
            continue

        m1, m2 = non_self[0], non_self[1]

        # Lowe's ratio test
        if m1.distance < ratio_threshold * m2.distance:
            idx1, idx2 = m1.queryIdx, m1.trainIdx
            pair_key = (min(idx1, idx2), max(idx1, idx2))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pt1 = keypoints[idx1].pt
            pt2 = keypoints[idx2].pt

            # Spatial distance filter (must be separated by min_distance)
            spatial_dist = float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))
            if spatial_dist >= min_distance:
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                valid_pairs.append({
                    "pt1": (int(round(pt1[0])), int(round(pt1[1]))),
                    "pt2": (int(round(pt2[0])), int(round(pt2[1]))),
                    "dist": spatial_dist,
                    "dx": dx,
                    "dy": dy
                })

    # Step 3: Spatial Displacement Vector Clustering
    # Duplicated patches share consistent translation vectors (dx, dy)
    num_clusters = 0
    consistent_pairs = []

    # Cap to top 300 candidate pairs to avoid quadratic stalls on repetitive textures
    if len(valid_pairs) > 300:
        valid_pairs = valid_pairs[:300]

    if len(valid_pairs) >= 3:
        vectors = np.array([[p["dx"], p["dy"]] for p in valid_pairs])
        visited = set()
        clusters = []

        for i in range(len(vectors)):
            if i in visited:
                continue
            current_cluster = [valid_pairs[i]]
            visited.add(i)
            for j in range(i + 1, len(vectors)):
                if j in visited:
                    continue
                v_diff = np.hypot(vectors[i, 0] - vectors[j, 0], vectors[i, 1] - vectors[j, 1])
                # Within 32 pixels translation variance
                if v_diff < 32.0:
                    current_cluster.append(valid_pairs[j])
                    visited.add(j)
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)

        num_clusters = len(clusters)
        for cl in clusters:
            consistent_pairs.extend(cl)

    # Use clustered pairs if found; otherwise retain filtered pairs
    active_matches = consistent_pairs if len(consistent_pairs) >= 3 else valid_pairs
    match_count = len(active_matches)

    # Step 4: Visualization
    line_color = (255, 50, 0)      # Neon Orange-Red
    src_pt_color = (0, 255, 100)   # Vivid Lime Green
    dst_pt_color = (30, 144, 255)  # Dodger Blue

    for item in active_matches:
        p1 = item["pt1"]
        p2 = item["pt2"]
        # Draw circles and linking lines
        cv2.circle(annotated, p1, 5, src_pt_color, -1)
        cv2.circle(annotated, p1, 7, (0, 0, 0), 1)
        cv2.circle(annotated, p2, 5, dst_pt_color, -1)
        cv2.circle(annotated, p2, 7, (0, 0, 0), 1)
        cv2.line(annotated, p1, p2, line_color, 2, cv2.LINE_AA)

    # Step 5: Confidence Calculation
    if match_count >= 8 or (match_count >= 4 and num_clusters >= 1):
        is_tampered = True
        confidence = min(99.0, 70.0 + (match_count * 2.0))
    elif match_count >= 3:
        is_tampered = True
        confidence = min(75.0, 50.0 + (match_count * 5.0))
    elif match_count >= 1:
        is_tampered = False
        confidence = 25.0
    else:
        is_tampered = False
        confidence = 5.0

    return {
        "annotated_image": annotated,
        "match_count": int(match_count),
        "is_tampered": bool(is_tampered),
        "confidence": round(float(confidence), 1),
        "clusters": int(num_clusters),
        "keypoints_found": int(len(keypoints))
    }
