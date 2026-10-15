"""
Image Forensic Analyzer - Synthetic & Demonstration Dataset Generator
-----------------------------------------------------------------------
Generates verified authentic and tampered (Copy-Move, Splicing)
photographic test images with ground truth annotations.

Designed for multimedia forensics benchmarks following standard
CASIA / MICC / CoMoFoD test protocols.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image


def create_natural_landscape(width=640, height=480, seed=42):
    """Creates a photorealistic synthetic natural landscape with gradients and sensor noise."""
    np.random.seed(seed)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(int(height * 0.65)):
        ratio = y / (height * 0.65)
        r = int(70 * (1 - ratio) + 240 * ratio)
        g = int(130 * (1 - ratio) + 180 * ratio)
        b = int(220 * (1 - ratio) + 110 * ratio)
        canvas[y, :] = [r, g, b]

    # Ground gradient
    for y in range(int(height * 0.65), height):
        ratio = (y - height * 0.65) / (height * 0.35)
        r = int(34 * (1 - ratio) + 20 * ratio)
        g = int(120 * (1 - ratio) + 60 * ratio)
        b = int(45 * (1 - ratio) + 25 * ratio)
        canvas[y, :] = [r, g, b]

    # Mountains
    pts1 = np.array([
        [0, int(height * 0.65)],
        [int(width * 0.2), int(height * 0.40)],
        [int(width * 0.45), int(height * 0.55)],
        [int(width * 0.7), int(height * 0.35)],
        [width, int(height * 0.60)],
        [width, height],
        [0, height]
    ], np.int32)
    cv2.fillPoly(canvas, [pts1], (60, 75, 95))

    # Sun
    cv2.circle(canvas, (int(width * 0.8), int(height * 0.25)), 35, (255, 235, 170), -1)

    # Windmill Tower (Distinctive texture for ORB)
    cx, cy = int(width * 0.3), int(height * 0.62)
    cv2.rectangle(canvas, (cx - 16, cy - 65), (cx + 16, cy + 15), (140, 100, 70), -1)
    roof_pts = np.array([[cx - 24, cy - 65], [cx + 24, cy - 65], [cx, cy - 105]], np.int32)
    cv2.fillPoly(canvas, [roof_pts], (180, 50, 40))
    for wy in range(cy - 50, cy, 14):
        cv2.rectangle(canvas, (cx - 8, wy), (cx + 8, wy + 8), (240, 240, 220), -1)
        cv2.rectangle(canvas, (cx - 8, wy), (cx + 8, wy + 8), (40, 40, 40), 1)

    # Blades
    cv2.circle(canvas, (cx, cy - 65), 5, (30, 30, 30), -1)
    blade_len = 38
    cv2.line(canvas, (cx, cy - 65), (cx + blade_len, cy - 65 + blade_len), (230, 230, 230), 4)
    cv2.line(canvas, (cx, cy - 65), (cx - blade_len, cy - 65 - blade_len), (230, 230, 230), 4)
    cv2.line(canvas, (cx, cy - 65), (cx - blade_len, cy - 65 + blade_len), (230, 230, 230), 4)
    cv2.line(canvas, (cx, cy - 65), (cx + blade_len, cy - 65 - blade_len), (230, 230, 230), 4)

    # Sensor noise
    sensor_noise = np.random.normal(0, 3.5, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)


def create_lake_scene(width=640, height=480, seed=101):
    """Creates a natural lake scene with mountains, reflective water, and a textured sailboat."""
    np.random.seed(seed)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(int(height * 0.5)):
        ratio = y / (height * 0.5)
        canvas[y, :] = [int(180 * (1 - ratio) + 230 * ratio),
                        int(190 * (1 - ratio) + 210 * ratio),
                        int(235 * (1 - ratio) + 180 * ratio)]

    # Lake Water
    for y in range(int(height * 0.5), height):
        ratio = (y - height * 0.5) / (height * 0.5)
        canvas[y, :] = [int(60 + 20 * ratio), int(110 + 25 * ratio), int(160 + 30 * ratio)]

    # Mountain ridges
    pts_mount = np.array([
        [0, int(height * 0.5)],
        [int(width * 0.25), int(height * 0.22)],
        [int(width * 0.5), int(height * 0.38)],
        [int(width * 0.75), int(height * 0.18)],
        [width, int(height * 0.5)],
        [width, height],
        [0, height]
    ], np.int32)
    cv2.fillPoly(canvas, [pts_mount], (75, 70, 85))

    # Single Sailboat (textured source for ORB features)
    # Hull
    cv2.rectangle(canvas, (370, 270), (470, 290), (120, 50, 30), -1)
    # Triangular Main Sail
    pts_sail = np.array([[380, 270], [460, 270], [420, 160]], np.int32)
    cv2.fillPoly(canvas, [pts_sail], (245, 245, 245))
    cv2.line(canvas, (420, 160), (420, 270), (40, 40, 40), 2)
    # Mast banner
    cv2.circle(canvas, (420, 155), 4, (200, 30, 30), -1)

    sensor_noise = np.random.normal(0, 3.2, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)


def generate_demonstration_dataset(base_dir="dataset"):
    """Builds authentic and tampered samples for the forensics benchmark."""
    auth_dir = os.path.join(base_dir, "authentic")
    cm_dir = os.path.join(base_dir, "tampered", "copy_move")
    splice_dir = os.path.join(base_dir, "tampered", "splicing")

    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(splice_dir, exist_ok=True)

    ground_truth = []

    # 1. Authentic Landscape
    land_img = create_natural_landscape(640, 480, seed=42)
    auth_land_path = os.path.join(auth_dir, "authentic_landscape.jpg")
    Image.fromarray(land_img).save(auth_land_path, format="JPEG", quality=95)
    print(f"[+] Created: {auth_land_path}")
    ground_truth.append({
        "filename": "authentic/authentic_landscape.jpg",
        "is_tampered": False,
        "tampering_type": "none",
        "description": "Original untouched natural landscape with uniform sensor noise."
    })

    # 2. Authentic Mountain Lake
    lake_img = create_lake_scene(640, 480, seed=101)
    auth_lake_path = os.path.join(auth_dir, "authentic_lake.jpg")
    Image.fromarray(lake_img).save(auth_lake_path, format="JPEG", quality=95)
    print(f"[+] Created: {auth_lake_path}")
    ground_truth.append({
        "filename": "authentic/authentic_lake.jpg",
        "is_tampered": False,
        "tampering_type": "none",
        "description": "Original untouched mountain lake scene with single sailboat."
    })

    # 3. Copy-Move: Duplicated Windmill Tower
    cm1_canvas = land_img.copy()
    src_x1, src_y1 = 155, 270
    src_x2, src_y2 = 235, 410
    patch1 = cm1_canvas[src_y1:src_y2, src_x1:src_x2].copy()
    dst_x1, dst_y1 = 430, 270
    h_p, w_p = patch1.shape[:2]
    cm1_canvas[dst_y1:dst_y1 + h_p, dst_x1:dst_x1 + w_p] = patch1

    cm_path1 = os.path.join(cm_dir, "forgery_copy_move_tower.jpg")
    Image.fromarray(cm1_canvas).save(cm_path1, format="JPEG", quality=95)
    print(f"[+] Created: {cm_path1}")
    ground_truth.append({
        "filename": "tampered/copy_move/forgery_copy_move_tower.jpg",
        "is_tampered": True,
        "tampering_type": "copy_move",
        "description": "Windmill tower duplicated to right side of landscape via copy-move."
    })

    # 4. Copy-Move: Duplicated Sailboat
    cm2_canvas = lake_img.copy()
    boat_patch = cm2_canvas[150:295, 360:480].copy()
    cm2_canvas[150:295, 120:240] = boat_patch

    cm_path2 = os.path.join(cm_dir, "forgery_copy_move_boat.jpg")
    Image.fromarray(cm2_canvas).save(cm_path2, format="JPEG", quality=95)
    print(f"[+] Created: {cm_path2}")
    ground_truth.append({
        "filename": "tampered/copy_move/forgery_copy_move_boat.jpg",
        "is_tampered": True,
        "tampering_type": "copy_move",
        "description": "Sailboat cloned across the lake surface to deceive vessel count."
    })

    # 5. Splicing: Foreign Aircraft/Drone inserted into Sky
    splice1_canvas = land_img.copy()
    pw, ph = 100, 60
    donor_patch = np.zeros((ph, pw, 3), dtype=np.uint8)
    donor_patch[:] = [210, 220, 230]
    donor_noise = np.random.normal(0, 18.0, donor_patch.shape).astype(np.int16)
    donor_patch = np.clip(donor_patch.astype(np.int16) + donor_noise, 0, 255).astype(np.uint8)
    cv2.ellipse(donor_patch, (50, 30), (42, 14), 0, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(donor_patch, (45, 10), (55, 30), (10, 10, 10), -1)

    sx, sy = 380, 70
    splice1_canvas[sy:sy + ph, sx:sx + pw] = donor_patch

    splice_path1 = os.path.join(splice_dir, "forgery_spliced_uav.jpg")
    Image.fromarray(splice1_canvas).save(splice_path1, format="JPEG", quality=82)
    print(f"[+] Created: {splice_path1}")
    ground_truth.append({
        "filename": "tampered/splicing/forgery_spliced_uav.jpg",
        "is_tampered": True,
        "tampering_type": "splicing",
        "description": "Foreign aerial vehicle spliced into sky with mismatched compression quality."
    })

    # 6. Splicing: Foreign Hot-Air Balloon spliced into Lake Sky
    splice2_canvas = lake_img.copy()
    bw, bh = 80, 100
    balloon = np.zeros((bh, bw, 3), dtype=np.uint8)
    balloon[:] = [230, 220, 210]
    # Red/yellow balloon stripes
    cv2.circle(balloon, (40, 45), 38, (220, 40, 30), -1)
    cv2.ellipse(balloon, (40, 45), (18, 38), 0, 0, 360, (250, 220, 40), -1)
    cv2.rectangle(balloon, (32, 85), (48, 95), (100, 60, 30), -1)
    b_noise = np.random.normal(0, 15.0, balloon.shape).astype(np.int16)
    balloon = np.clip(balloon.astype(np.int16) + b_noise, 0, 255).astype(np.uint8)

    splice2_canvas[40:40 + bh, 200:200 + bw] = balloon
    splice_path2 = os.path.join(splice_dir, "forgery_spliced_balloon.jpg")
    Image.fromarray(splice2_canvas).save(splice_path2, format="JPEG", quality=80)
    print(f"[+] Created: {splice_path2}")
    ground_truth.append({
        "filename": "tampered/splicing/forgery_spliced_balloon.jpg",
        "is_tampered": True,
        "tampering_type": "splicing",
        "description": "Foreign hot-air balloon spliced into mountain sky with anomalous noise."
    })

    # Save Ground Truth Manifest
    gt_file = os.path.join(base_dir, "ground_truth.json")
    with open(gt_file, "w") as f:
        json.dump(ground_truth, f, indent=4)
    print(f"[+] Ground truth manifest saved to {gt_file}")

    return ground_truth


if __name__ == "__main__":
    generate_demonstration_dataset()
