"""
Image Forensic Analyzer - Tkinter Desktop Graphical User Interface
-------------------------------------------------------------------
A standalone native desktop GUI application for digital tampering detection.
Built using Python's standard Tkinter and Pillow libraries.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.utils import (
    validate_image,
    load_image_rgb,
    compute_image_statistics,
    calculate_overall_verdict
)
from src.ela import perform_ela
from src.copy_move import detect_copy_move
from src.splicing_detection import detect_splicing_noise
from src.metadata_analyzer import extract_metadata


class ForensicAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Forensic Analyzer - Digital Tampering Detection")
        self.root.geometry("1100x820")
        self.root.minsize(950, 700)
        self.root.configure(bg="#1E1E2F")

        self.current_image_path = None
        self.pil_original = None
        self.np_original = None

        # Build UI Components
        self.create_widgets()

    def create_widgets(self):
        # 1. Header Frame
        header_frame = tk.Frame(self.root, bg="#27293D", height=70)
        header_frame.pack(fill="x", padx=10, pady=5)

        title_label = tk.Label(
            header_frame,
            text="🔍 IMAGE FORENSIC ANALYZER",
            font=("Segoe UI", 18, "bold"),
            fg="#00E676",
            bg="#27293D"
        )
        title_label.pack(side="left", padx=20, pady=10)

        subtitle_label = tk.Label(
            header_frame,
            text="Digital Tampering & Forgery Detection Lab",
            font=("Segoe UI", 10),
            fg="#8BE9FD",
            bg="#27293D"
        )
        subtitle_label.pack(side="left", padx=5, pady=12)

        # 2. Controls Toolbar
        toolbar = tk.Frame(self.root, bg="#1E1E2F")
        toolbar.pack(fill="x", padx=20, pady=8)

        btn_browse = tk.Button(
            toolbar,
            text="📁 Browse Image...",
            font=("Segoe UI", 10, "bold"),
            bg="#2196F3",
            fg="white",
            activebackground="#1976D2",
            padx=14,
            pady=6,
            relief="flat",
            command=self.browse_image
        )
        btn_browse.pack(side="left", padx=5)

        btn_sample = tk.Button(
            toolbar,
            text="🧪 Load Demo Forgery Sample",
            font=("Segoe UI", 10),
            bg="#3F51B5",
            fg="white",
            activebackground="#303F9F",
            padx=14,
            pady=6,
            relief="flat",
            command=self.load_demo_sample
        )
        btn_sample.pack(side="left", padx=5)

        btn_analyze = tk.Button(
            toolbar,
            text="🔬 ANALYZE EVIDENCE",
            font=("Segoe UI", 10, "bold"),
            bg="#00C853",
            fg="white",
            activebackground="#009624",
            padx=18,
            pady=6,
            relief="flat",
            command=self.analyze_image
        )
        btn_analyze.pack(side="left", padx=15)

        self.lbl_path = tk.Label(
            toolbar,
            text="No image loaded",
            font=("Segoe UI", 9, "italic"),
            fg="#B0BEC5",
            bg="#1E1E2F"
        )
        self.lbl_path.pack(side="left", padx=10)

        # 3. Verdict Banner
        self.verdict_frame = tk.Frame(self.root, bg="#37474F", height=60)
        self.verdict_frame.pack(fill="x", padx=20, pady=5)

        self.lbl_verdict = tk.Label(
            self.verdict_frame,
            text="AWAITING IMAGE EVIDENCE TO ANALYZE",
            font=("Segoe UI", 14, "bold"),
            fg="#FFFFFF",
            bg="#37474F"
        )
        self.lbl_verdict.pack(pady=8)

        # 4. Image Display Panels (3 side-by-side columns: Original, ELA, Copy-Move)
        display_frame = tk.Frame(self.root, bg="#1E1E2F")
        display_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Panel 1: Original Image
        p1 = tk.LabelFrame(display_frame, text="1. Original Target Image", font=("Segoe UI", 10, "bold"), fg="#82AAFF", bg="#27293D", padx=5, pady=5)
        p1.pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_orig = tk.Label(p1, bg="#1E1E2F", text="Original preview")
        self.canvas_orig.pack(fill="both", expand=True)

        # Panel 2: ELA Enhanced
        p2 = tk.LabelFrame(display_frame, text="2. Error Level Analysis (ELA)", font=("Segoe UI", 10, "bold"), fg="#FFCB6B", bg="#27293D", padx=5, pady=5)
        p2.pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_ela = tk.Label(p2, bg="#1E1E2F", text="ELA preview")
        self.canvas_ela.pack(fill="both", expand=True)

        # Panel 3: Copy-Move Matches
        p3 = tk.LabelFrame(display_frame, text="3. Copy-Move Forgery Points", font=("Segoe UI", 10, "bold"), fg="#F07178", bg="#27293D", padx=5, pady=5)
        p3.pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_cm = tk.Label(p3, bg="#1E1E2F", text="Copy-Move preview")
        self.canvas_cm.pack(fill="both", expand=True)

        # 5. Diagnostic Log / Forensic Metrics Console
        metrics_frame = tk.LabelFrame(
            self.root,
            text="📋 Detailed Forensic Findings & Metric Breakdown",
            font=("Segoe UI", 10, "bold"),
            fg="#C3E88D",
            bg="#27293D",
            padx=10,
            pady=5
        )
        metrics_frame.pack(fill="x", padx=20, pady=10)

        self.txt_log = tk.Text(
            metrics_frame,
            height=6,
            bg="#181824",
            fg="#EEFFFF",
            font=("Consolas", 10),
            relief="flat",
            wrap="word"
        )
        self.txt_log.pack(fill="both", expand=True, padx=5, pady=5)
        self.txt_log.insert("end", "System initialized. Load an image and click 'ANALYZE EVIDENCE' to begin.")
        self.txt_log.config(state="disabled")

    def resize_for_preview(self, pil_img, max_w=320, max_h=240):
        """Resizes a PIL image maintaining aspect ratio for panel display."""
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.tiff *.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_image_path(file_path)

    def load_demo_sample(self):
        sample_path = "dataset/tampered/copy_move/forgery_copy_move_tower.jpg"
        if not os.path.exists(sample_path):
            from generate_sample_data import generate_demonstration_dataset
            generate_demonstration_dataset()
        self.load_image_path(sample_path)

    def load_image_path(self, path):
        self.current_image_path = path
        self.lbl_path.config(text=os.path.basename(path))

        is_valid, err_msg, pil_img, np_img = validate_image(path)
        if not is_valid:
            messagebox.showerror("Invalid Image", f"Cannot load image:\n{err_msg}")
            return

        try:
            self.pil_original, self.np_original = pil_img, np_img
            self.tk_orig = self.resize_for_preview(self.pil_original)
            self.canvas_orig.config(image=self.tk_orig, text="")

            # Reset other preview panels
            self.canvas_ela.config(image="", text="Click Analyze")
            self.canvas_cm.config(image="", text="Click Analyze")

            self.lbl_verdict.config(
                text="IMAGE LOADED • READY FOR ANALYSIS",
                bg="#37474F",
                fg="#FFFFFF"
            )
            self.verdict_frame.config(bg="#37474F")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def analyze_image(self):
        if self.current_image_path is None or self.np_original is None:
            messagebox.showwarning("Warning", "Please select or browse an image first!")
            return

        try:
            # 1. Image Statistics
            stats = compute_image_statistics(self.np_original)
            # 2. EXIF Metadata
            meta = extract_metadata(self.current_image_path)
            # 3. ELA
            ela_res = perform_ela(self.np_original, quality=90, scale_factor=15)
            # 4. Copy-Move
            cm_res = detect_copy_move(self.np_original, max_features=3000)
            # 5. Splicing / Noise
            splice_res = detect_splicing_noise(self.np_original)
            # 6. Overall Decision Fusion
            verdict = calculate_overall_verdict(
                ela_score=ela_res["ela_tampering_score"],
                copy_move_matches=cm_res["match_count"],
                noise_score=splice_res["noise_inconsistency_score"],
                exif_suspicious=meta["is_suspicious"]
            )

            # Update ELA Preview Panel
            pil_ela = Image.fromarray(ela_res["ela_image_np"])
            self.tk_ela = self.resize_for_preview(pil_ela)
            self.canvas_ela.config(image=self.tk_ela, text="")

            # Update Copy-Move Preview Panel
            pil_cm = Image.fromarray(cm_res["annotated_image"])
            self.tk_cm = self.resize_for_preview(pil_cm)
            self.canvas_cm.config(image=self.tk_cm, text="")

            # Update Verdict Banner
            if verdict["risk_level"] == "HIGH":
                v_bg = "#D32F2F"
            elif verdict["risk_level"] == "MODERATE":
                v_bg = "#F57C00"
            else:
                v_bg = "#388E3C"

            self.lbl_verdict.config(
                text=f"VERDICT: {verdict['verdict']}   |   CONFIDENCE: {verdict['confidence_score']}%",
                bg=v_bg,
                fg="#FFFFFF"
            )
            self.verdict_frame.config(bg=v_bg)

            # Populate Metrics Log
            log_text = (
                f"File: {os.path.basename(self.current_image_path)} | Size: {stats['width']}x{stats['height']} px | Entropy: {stats['entropy']} bits\n"
                f"----------------------------------------------------------------------------------------------------\n"
                f"• Verdict: {verdict['verdict']} (Confidence: {verdict['confidence_score']}%, Risk: {verdict['risk_level']})\n"
                f"• Primary Indicator: {verdict['primary_technique']}\n"
                f"• Copy-Move: {cm_res['match_count']} duplicated keypoint pairs detected across {cm_res['clusters']} spatial clusters\n"
                f"• Error Level Analysis (ELA): Score {ela_res['ela_tampering_score']}/100 | Mean Error: {ela_res['mean_error']} | Outliers: {ela_res['high_error_ratio']}%\n"
                f"• Noise Inconsistency / Splicing: P95/Med Ratio: {splice_res['variance_ratio']}x (Normal < 3.0x)\n"
                f"• EXIF Metadata: {'[ALERT] ' + '; '.join(meta['reasons']) if meta['is_suspicious'] else 'No software tampering detected in headers.'}\n"
            )

            self.txt_log.config(state="normal")
            self.txt_log.delete("1.0", "end")
            self.txt_log.insert("end", log_text)
            self.txt_log.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during forensic execution:\n{e}")


def main():
    root = tk.Tk()
    app = ForensicAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
