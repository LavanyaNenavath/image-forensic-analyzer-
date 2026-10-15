"""
Image Forensic Analyzer - Performance Evaluation Module
---------------------------------------------------------
Calculates standard forensic evaluation metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Specificity
- Confusion Matrix

Validates algorithm performance across authentic and tampered datasets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_image_rgb, calculate_overall_verdict
from src.ela import perform_ela
from src.copy_move import detect_copy_move
from src.splicing_detection import detect_splicing_noise
from src.metadata_analyzer import extract_metadata


def evaluate_dataset(dataset_dir="dataset", ground_truth_file=None, save_matrix_path="reports/confusion_matrix.png"):
    """
    Evaluates the image forensics pipeline on the dataset and outputs metrics.
    """
    if ground_truth_file is None:
        ground_truth_file = os.path.join(dataset_dir, "ground_truth.json")

    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    with open(ground_truth_file, "r") as f:
        records = json.load(f)

    y_true = []
    y_pred = []
    results_table = []

    print("=" * 75)
    print("      IMAGE FORENSIC ANALYZER - PERFORMANCE EVALUATION BENCHMARK      ")
    print("=" * 75)

    for item in records:
        rel_path = item["filename"]
        img_path = os.path.join(dataset_dir, rel_path)
        actual_tampered = item["is_tampered"]

        if not os.path.exists(img_path):
            print(f"[!] Warning: File {img_path} not found. Skipping.")
            continue

        # Load image
        pil_img, np_img = load_image_rgb(img_path)

        # 1. Run ELA
        ela_res = perform_ela(np_img, quality=90, scale_factor=15)
        # 2. Run Copy-Move
        cm_res = detect_copy_move(np_img, max_features=2500)
        # 3. Run Splicing / Noise
        splice_res = detect_splicing_noise(np_img)
        # 4. Run Metadata
        meta_res = extract_metadata(img_path)

        # Synthesize Overall Verdict
        verdict_res = calculate_overall_verdict(
            ela_score=ela_res["ela_tampering_score"],
            copy_move_matches=cm_res["match_count"],
            noise_score=splice_res["noise_inconsistency_score"],
            exif_suspicious=meta_res["is_suspicious"]
        )

        pred_tampered = (verdict_res["risk_level"] in ["HIGH", "MODERATE"])

        y_true.append(1 if actual_tampered else 0)
        y_pred.append(1 if pred_tampered else 0)

        results_table.append({
            "filename": os.path.basename(rel_path),
            "actual": "TAMPERED" if actual_tampered else "AUTHENTIC",
            "predicted": verdict_res["verdict"],
            "confidence": f"{verdict_res['confidence_score']}%",
            "matches": cm_res["match_count"],
            "ela_score": ela_res["ela_tampering_score"],
            "correct": (actual_tampered == pred_tampered)
        })

    # Confusion Matrix Counts
    # TP: Actual Tampered (1), Predicted Tampered (1)
    # TN: Actual Authentic (0), Predicted Authentic (0)
    # FP: Actual Authentic (0), Predicted Tampered (1)
    # FN: Actual Tampered (1), Predicted Authentic (0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nTotal Test Images Evaluated: {total}")
    print("-" * 75)
    print(f"{'Filename':<32} | {'Actual':<10} | {'Confidence':<10} | {'Result':<8}")
    print("-" * 75)
    for r in results_table:
        status_sym = "[PASS]" if r["correct"] else "[FAIL]"
        print(f"{r['filename']:<32} | {r['actual']:<10} | {r['confidence']:<10} | {status_sym:<8}")
    print("-" * 75)

    print("\n" + "=" * 45)
    print("         PERFORMANCE METRICS SUMMARY         ")
    print("=" * 45)
    print(f" True Positives  (TP) : {tp}")
    print(f" True Negatives  (TN) : {tn}")
    print(f" False Positives (FP) : {fp}")
    print(f" False Negatives (FN) : {fn}")
    print("-" * 45)
    print(f" Accuracy            : {accuracy * 100:.2f}%")
    print(f" Precision           : {precision * 100:.2f}%")
    print(f" Recall (Sensitivity): {recall * 100:.2f}%")
    print(f" Specificity         : {specificity * 100:.2f}%")
    print(f" F1-Score            : {f1_score * 100:.2f}%")
    print("=" * 45)

    # Plot Confusion Matrix
    os.makedirs(os.path.dirname(save_matrix_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    matrix = np.array([[tn, fp], [fn, tp]])

    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.text(j, i, str(val), va='center', ha='center', fontsize=16, fontweight='bold',
                    color='white' if val > (matrix.max() / 2) else 'black')

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Authentic', 'Tampered'], fontsize=11, fontweight='bold')
    ax.set_yticklabels(['Authentic', 'Tampered'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Actual Ground Truth', fontsize=12, fontweight='bold')
    ax.set_title('Image Forensics - Confusion Matrix', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_matrix_path)
    plt.close()
    print(f"\n[+] Confusion Matrix plot saved to: {save_matrix_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }


if __name__ == "__main__":
    evaluate_dataset()
