"""
Image Forensic Analyzer - Automated Verification & Test Suite
--------------------------------------------------------------
Executes comprehensive unit and integration tests across all forensic modules:
- ELA Module
- Copy-Move Forgery Detector (ORB)
- Splicing & Noise Residual Module
- EXIF Metadata Auditor & Header Forensics
- Utility, Validation, & Decision Fusion Engine
- Uploaded File Handling (BytesIO/UploadedFile simulation)
- Cross-Format Testing (JPG, JPEG, PNG, WEBP, BMP, TIFF)
"""

import io
import os
import unittest
import numpy as np
from PIL import Image, PngImagePlugin

from src.utils import (
    validate_image,
    load_image_rgb,
    compute_image_statistics,
    create_heatmap_overlay,
    calculate_overall_verdict
)
from src.ela import perform_ela, analyze_ela_regions
from src.copy_move import detect_copy_move
from src.splicing_detection import detect_splicing_noise
from src.metadata_analyzer import extract_metadata


class MockUploadedFile(io.BytesIO):
    """Simulates Streamlit's UploadedFile object."""
    def __init__(self, data, name="uploaded_test.jpg"):
        super().__init__(data)
        self.name = name
        self.size = len(data)


class TestImageForensicAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure sample dataset is present for integration tests."""
        from generate_sample_data import generate_demonstration_dataset
        generate_demonstration_dataset()

    def test_utils_image_loading(self):
        """Verifies loading image as both PIL Image and NumPy RGB array."""
        test_path = "dataset/authentic/authentic_landscape.jpg"
        pil_img, np_img = load_image_rgb(test_path)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertIsInstance(np_img, np.ndarray)
        self.assertEqual(np_img.shape[2], 3)
        self.assertEqual(np_img.dtype, np.uint8)

    def test_utils_image_statistics(self):
        """Verifies mathematical statistics calculation (mean, std, entropy, sharpness)."""
        test_path = "dataset/authentic/authentic_landscape.jpg"
        _, np_img = load_image_rgb(test_path)
        stats = compute_image_statistics(np_img)
        self.assertIn("mean_brightness", stats)
        self.assertIn("std_deviation", stats)
        self.assertIn("entropy", stats)
        self.assertIn("sharpness_var", stats)
        self.assertGreater(stats["entropy"], 0)

    def test_ela_execution(self):
        """Verifies ELA produces valid enhanced difference and heatmap images."""
        test_path = "dataset/authentic/authentic_landscape.jpg"
        _, np_img = load_image_rgb(test_path)
        res = perform_ela(np_img, quality=90, scale_factor=15)
        self.assertIn("ela_image_np", res)
        self.assertIn("heatmap_rgb", res)
        self.assertIn("ela_tampering_score", res)
        self.assertGreaterEqual(res["ela_tampering_score"], 0)
        self.assertLessEqual(res["ela_tampering_score"], 100)

    def test_copy_move_detection(self):
        """Verifies Copy-Move detector identifies duplicated tower in tampered image."""
        cm_path = "dataset/tampered/copy_move/forgery_copy_move_tower.jpg"
        _, np_img = load_image_rgb(cm_path)
        res = detect_copy_move(np_img, max_features=3000)
        self.assertIn("match_count", res)
        self.assertIn("annotated_image", res)
        self.assertGreaterEqual(res["match_count"], 4)
        self.assertTrue(res["is_tampered"])

    def test_splicing_detection(self):
        """Verifies Splicing detector flags high noise variance ratio on spliced image."""
        splice_path = "dataset/tampered/splicing/forgery_spliced_uav.jpg"
        _, np_img = load_image_rgb(splice_path)
        res = detect_splicing_noise(np_img)
        self.assertIn("variance_ratio", res)
        self.assertIn("noise_inconsistency_score", res)
        self.assertGreaterEqual(res["variance_ratio"], 5.0)
        self.assertTrue(res["is_splicing_detected"])

    def test_decision_fusion(self):
        """Verifies decision fusion correctly assigns HIGH risk to positive indicators."""
        verdict_tampered = calculate_overall_verdict(
            ela_score=15.0, copy_move_matches=14, noise_score=10.0, exif_suspicious=False
        )
        self.assertEqual(verdict_tampered["risk_level"], "HIGH")
        self.assertIn("POTENTIAL TAMPERING", verdict_tampered["verdict"])

        verdict_authentic = calculate_overall_verdict(
            ela_score=5.0, copy_move_matches=0, noise_score=10.0, exif_suspicious=False
        )
        self.assertEqual(verdict_authentic["risk_level"], "LOW")
        self.assertIn("NO SIGNIFICANT FORENSIC INDICATORS", verdict_authentic["verdict"])

    def test_uploaded_file_stream_handling(self):
        """
        Verifies that Streamlit UploadedFile simulation (BytesIO stream)
        can be validated, loaded, and analyzed without throwing:
        'AttributeError: UploadedFile object has no attribute getexif'.
        """
        raw_img = Image.new("RGB", (200, 200), color=(120, 150, 180))
        buf = io.BytesIO()
        raw_img.save(buf, format="JPEG")
        mock_file = MockUploadedFile(buf.getvalue(), "user_upload.jpg")

        # 1. Validation
        is_valid, err, pil_img, np_img = validate_image(mock_file)
        self.assertTrue(is_valid, f"Validation failed: {err}")

        # 2. Metadata extraction directly on uploaded file object
        meta = extract_metadata(mock_file)
        self.assertIsInstance(meta, dict)
        self.assertIn("exif_found", meta)
        self.assertIn("tags", meta)
        self.assertEqual(meta["file_format"], "JPEG")

        # 3. Complete forensic pipeline execution
        ela = perform_ela(np_img)
        cm = detect_copy_move(np_img)
        splice = detect_splicing_noise(np_img)
        verdict = calculate_overall_verdict(
            ela["ela_tampering_score"],
            cm["match_count"],
            splice["noise_inconsistency_score"],
            meta["is_suspicious"]
        )
        self.assertIn("verdict", verdict)
        self.assertIn("forensic_risk_score", verdict)

    def test_all_supported_formats(self):
        """
        Verifies full forensic pipeline works seamlessly across all required formats:
        JPG, JPEG, PNG, WEBP, BMP, and TIFF.
        """
        formats_to_test = [
            ("JPEG", "test.jpg"),
            ("JPEG", "test.jpeg"),
            ("PNG", "test.png"),
            ("WEBP", "test.webp"),
            ("BMP", "test.bmp"),
            ("TIFF", "test.tiff")
        ]

        test_pattern = np.zeros((120, 120, 3), dtype=np.uint8)
        # Add synthetic textural pattern so ORB and ELA have signal
        test_pattern[20:60, 20:60] = [200, 100, 50]
        test_pattern[70:100, 70:100] = [50, 150, 200]
        base_pil = Image.fromarray(test_pattern)

        for fmt, fname in formats_to_test:
            with self.subTest(format=fmt, filename=fname):
                buf = io.BytesIO()
                base_pil.save(buf, format=fmt)
                mock_upload = MockUploadedFile(buf.getvalue(), fname)

                # Validation
                is_valid, err, pil_img, np_img = validate_image(mock_upload)
                self.assertTrue(is_valid, f"Failed validation for {fmt}: {err}")
                self.assertEqual(np_img.shape[2], 3)

                # Metadata
                meta = extract_metadata(mock_upload)
                self.assertIsInstance(meta, dict)

                # ELA
                ela_res = perform_ela(np_img)
                self.assertIn("ela_tampering_score", ela_res)

                # Copy-Move
                cm_res = detect_copy_move(np_img)
                self.assertIn("match_count", cm_res)

                # Splicing
                sp_res = detect_splicing_noise(np_img)
                self.assertIn("noise_inconsistency_score", sp_res)

                # Verdict
                verdict = calculate_overall_verdict(
                    ela_res["ela_tampering_score"],
                    cm_res["match_count"],
                    sp_res["noise_inconsistency_score"],
                    meta["is_suspicious"]
                )
                self.assertIn("verdict", verdict)

    def test_transparent_png_handling(self):
        """Verifies RGBA images with transparency composite cleanly over white without crash."""
        rgba_arr = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba_arr[:, :, 0] = 255  # Red
        rgba_arr[:, :, 3] = 128  # 50% opacity
        rgba_pil = Image.fromarray(rgba_arr, mode="RGBA")

        buf = io.BytesIO()
        rgba_pil.save(buf, format="PNG")
        mock_file = MockUploadedFile(buf.getvalue(), "transparent.png")

        is_valid, err, pil_img, np_img = validate_image(mock_file)
        self.assertTrue(is_valid)
        self.assertEqual(np_img.shape, (100, 100, 3))
        self.assertEqual(np_img.dtype, np.uint8)

        # Run ELA and noise analysis
        ela = perform_ela(np_img)
        self.assertIn("ela_tampering_score", ela)
        sp = detect_splicing_noise(np_img)
        self.assertIn("variance_ratio", sp)

    def test_missing_exif_handling(self):
        """Verifies that an image without EXIF (social media/screenshot) does not crash."""
        plain_pil = Image.new("RGB", (80, 80), color=(50, 100, 150))
        meta = extract_metadata(plain_pil)
        self.assertFalse(meta["exif_found"])
        self.assertFalse(meta["is_suspicious"])
        self.assertIsNone(meta["software_detected"])

    def test_suspicious_software_detection(self):
        """Verifies detecting editing signatures in PNG info chunk and EXIF."""
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        pinfo = PngImagePlugin.PngInfo()
        pinfo.add_text("Software", "Adobe Photoshop 2024")
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=pinfo)
        mock_file = MockUploadedFile(buf.getvalue(), "photoshop_export.png")

        meta = extract_metadata(mock_file)
        self.assertTrue(meta["is_suspicious"])
        self.assertIn("Adobe Photoshop 2024", meta["software_detected"])

    def test_corrupted_image_graceful_handling(self):
        """Verifies empty or corrupted image files are caught gracefully by validator."""
        # 1. Empty buffer
        empty_file = MockUploadedFile(b"", "empty.jpg")
        is_valid, err, _, _ = validate_image(empty_file)
        self.assertFalse(is_valid)
        self.assertIn("empty", err.lower())

        # 2. Corrupted random text bytes
        corrupted_file = MockUploadedFile(b"Not an image file at all! Just plain text.", "fake.png")
        is_valid, err, _, _ = validate_image(corrupted_file)
        self.assertFalse(is_valid)
        self.assertIn("corrupted", err.lower())

    def test_small_image_edge_case(self):
        """Verifies algorithms handle small dimension images (e.g. 24x24 px) without crash."""
        small_arr = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        is_valid, _, pil_img, np_img = validate_image(small_arr)
        self.assertTrue(is_valid)

        # Splicing noise must not raise IndexError on small images
        sp = detect_splicing_noise(np_img)
        self.assertIn("variance_ratio", sp)
        self.assertFalse(np.isnan(sp["variance_ratio"]))

        # ELA must run without error
        ela = perform_ela(np_img)
        self.assertIn("ela_tampering_score", ela)


if __name__ == "__main__":
    unittest.main()
