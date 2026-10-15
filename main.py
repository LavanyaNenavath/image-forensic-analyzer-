"""
Image Forensic Analyzer for Digital Tampering Detection
--------------------------------------------------------
Main CLI Application & Pipeline Orchestrator.

Usage:
    Analyze a single image:
        python main.py --image dataset/tampered/copy_move/forgery_copy_move_tower.jpg

    Run dataset performance benchmark:
        python main.py --evaluate

    Launch Streamlit Interactive Forensic Web App:
        python main.py --gui

    Launch Tkinter Native Desktop Forensic GUI:
        python main.py --tkinter

    Regenerate sample authentic/tampered test data:
        python main.py --generate-data
"""

import os
import sys
import argparse
import json
import subprocess
import cv2
import matplotlib.pyplot as plt

from src.utils import (
    validate_image,
    load_image_rgb,
    compute_image_statistics,
    create_heatmap_overlay,
    calculate_overall_verdict
)
from src.ela import perform_ela
from src.copy_move import detect_copy_move
from src.splicing_detection import detect_splicing_noise
from src.metadata_analyzer import extract_metadata


def analyze_single_image(image_path, output_dir="reports/analysis_output", show_plot=False):
    """
    Executes the complete multi-modal forensics pipeline on an image:
    1. EXIF metadata inspection
    2. Image statistics computation
    3. Error Level Analysis (ELA)
    4. Copy-Move Forgery Detection (ORB)
    5. Image Splicing / Noise Inconsistency Analysis
    6. Forensic Decision Fusion & Report Generation
    """
    is_valid, err_msg, pil_img, np_img = validate_image(image_path)
    if not is_valid:
        print(f"[!] Validation Error: {err_msg}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    print("\n" + "=" * 75)
    print(f"       ANALYZING DIGITAL EVIDENCE: {filename}       ")
    print("=" * 75)

    # 1. Image Statistics
    stats = compute_image_statistics(np_img)
    print(f"[*] Image Dimensions : {stats['width']}x{stats['height']} px ({stats['channels']} channels)")
    print(f"[*] Brightness / Std : Mean {stats['mean_brightness']}, Std {stats['std_deviation']}")
    print(f"[*] Sharpness Var    : {stats['sharpness_var']}")
    print(f"[*] Shannon Entropy  : {stats['entropy']} bits")

    # 2. Metadata Audit
    print("\n[+] Step 1: Performing EXIF & Digital Header Audit...")
    meta = extract_metadata(image_path)
    if meta["is_suspicious"]:
        print(f"    [!] ALERT: Suspicious metadata detected!")
        for r in meta["reasons"]:
            print(f"        - {r}")
    else:
        print("    [OK] No editing software signatures found in EXIF.")

    # 3. Error Level Analysis (ELA)
    print("\n[+] Step 2: Running Error Level Analysis (ELA @ Q=90)...")
    ela_res = perform_ela(np_img, quality=90, scale_factor=15)
    print(f"    - Mean Error        : {ela_res['mean_error']}")
    print(f"    - Error Std Dev     : {ela_res['std_error']}")
    print(f"    - High-Error Pixels : {ela_res['high_error_ratio']}%")
    print(f"    - ELA Anomaly Score : {ela_res['ela_tampering_score']} / 100")

    # 4. Copy-Move Forgery Detection
    print("\n[+] Step 3: Scanning for Copy-Move Forgeries (ORB Descriptors)...")
    cm_res = detect_copy_move(np_img, max_features=3000)
    print(f"    - Keypoints Checked : {cm_res['keypoints_found']}")
    print(f"    - Duplicate Matches : {cm_res['match_count']} pairs")
    print(f"    - Spatial Clusters  : {cm_res['clusters']}")
    print(f"    - Copy-Move Verdict : {'FORGERY FOUND' if cm_res['is_tampered'] else 'NO CLONES'}")

    # 5. Splicing & Noise Inconsistency Detection
    print("\n[+] Step 4: Assessing High-Frequency Noise & Splicing Residuals...")
    splice_res = detect_splicing_noise(np_img)
    print(f"    - Noise P95/Med Ratio: {splice_res['variance_ratio']}")
    print(f"    - Inconsistency Score: {splice_res['noise_inconsistency_score']} / 100")
    print(f"    - Noise Consistency  : {'POTENTIAL INCONSISTENCY' if splice_res['is_splicing_detected'] else 'CONSISTENT NOISE'}")

    # 6. Overall Multi-Modal Decision Fusion
    verdict_info = calculate_overall_verdict(
        ela_score=ela_res["ela_tampering_score"],
        copy_move_matches=cm_res["match_count"],
        noise_score=splice_res["noise_inconsistency_score"],
        exif_suspicious=meta["is_suspicious"]
    )

    print("\n" + "=" * 75)
    print("                      FINAL FORENSIC VERDICT                      ")
    print("=" * 75)
    print(f" STATUS VERDICT       : {verdict_info['verdict']}")
    print(f" RISK LEVEL           : {verdict_info['risk_level']}")
    print(f" FORENSIC RISK SCORE  : {verdict_info['forensic_risk_score']}%")
    print(f" PRIMARY TECHNIQUE    : {verdict_info['primary_technique']}")
    print("=" * 75)

    # 7. Generate Multi-Panel Forensic Evidence Sheet
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    fig.suptitle(f"Forensic Report: {filename}\nVerdict: {verdict_info['verdict']} ({verdict_info['forensic_risk_score']}%)",
                 fontsize=14, fontweight='bold', color=verdict_info['status_color'])

    # Panel 1: Original Image
    axes[0, 0].imshow(np_img)
    axes[0, 0].set_title("1. Original Image Under Test", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: ELA Difference Map
    axes[0, 1].imshow(ela_res['ela_image_np'])
    axes[0, 1].set_title(f"2. Error Level Analysis (Score: {ela_res['ela_tampering_score']})",
                         fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: Copy-Move Matches
    axes[1, 0].imshow(cm_res['annotated_image'])
    axes[1, 0].set_title(f"3. Copy-Move ORB Matches ({cm_res['match_count']} pairs)",
                         fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # Panel 4: Noise Variance Splicing Heatmap
    axes[1, 1].imshow(splice_res['variance_heatmap_rgb'])
    axes[1, 1].set_title(f"4. Noise Splicing Map (Ratio: {splice_res['variance_ratio']})",
                         fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    report_image_path = os.path.join(output_dir, f"{base_name}_forensic_panel.png")
    plt.savefig(report_image_path)
    if show_plot:
        plt.show()
    plt.close()

    print(f"\n[+] Forensic visual report saved to: {report_image_path}")

    # Save JSON Report
    report_data = {
        "filename": filename,
        "statistics": stats,
        "metadata": meta,
        "ela_analysis": {
            "mean_error": ela_res["mean_error"],
            "std_error": ela_res["std_error"],
            "high_error_ratio": ela_res["high_error_ratio"],
            "score": ela_res["ela_tampering_score"]
        },
        "copy_move_analysis": {
            "matches": cm_res["match_count"],
            "clusters": cm_res["clusters"],
            "is_tampered": cm_res["is_tampered"],
            "confidence": cm_res["confidence"]
        },
        "splicing_analysis": {
            "variance_ratio": splice_res["variance_ratio"],
            "score": splice_res["noise_inconsistency_score"],
            "is_splicing_detected": splice_res["is_splicing_detected"]
        },
        "final_verdict": verdict_info
    }

    report_json_path = os.path.join(output_dir, f"{base_name}_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"[+] Forensic JSON summary saved to : {report_json_path}\n")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Image Forensic Analyzer for Digital Tampering Detection")
    parser.add_argument("--image", type=str, help="Path to a single image to analyze")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark evaluation across the dataset")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic authentic & tampered datasets")
    parser.add_argument("--gui", action="store_true", help="Launch Streamlit interactive web application")
    parser.add_argument("--tkinter", action="store_true", help="Launch Tkinter desktop graphical interface")

    args = parser.parse_args()

    if args.generate_data:
        from generate_sample_data import generate_demonstration_dataset
        generate_demonstration_dataset()
    elif args.evaluate:
        from evaluate import evaluate_dataset
        evaluate_dataset()
    elif args.gui:
        print("[+] Launching Streamlit Forensic Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif args.tkinter:
        print("[+] Launching Tkinter Desktop Application...")
        subprocess.run([sys.executable, "gui_tkinter.py"])
    elif args.image:
        analyze_single_image(args.image)
    else:
        # Default behavior: run demonstration on authentic and tampered samples
        print("=" * 75)
        print("          IMAGE FORENSIC ANALYZER - DIGITAL TAMPERING DETECTION           ")
        print("=" * 75)
        print("No specific arguments provided. Available commands:")
        print("  python main.py --gui           : Launch interactive Streamlit Forensic Dashboard")
        print("  python main.py --tkinter       : Launch native desktop Tkinter GUI")
        print("  python main.py --evaluate      : Run performance metrics benchmark")
        print("  python main.py --image <path>  : Analyze an individual image")
        print("  python main.py --generate-data : Regenerate authentic and tampered samples")
        print("\nExecuting quick demo analysis on sample copy-move image...")

        sample_img = "dataset/tampered/copy_move/forgery_copy_move_tower.jpg"
        if not os.path.exists(sample_img):
            from generate_sample_data import generate_demonstration_dataset
            generate_demonstration_dataset()
        analyze_single_image(sample_img)


if __name__ == "__main__":
    main()
