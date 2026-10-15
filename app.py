"""
Image Forensic Analyzer - Streamlit Interactive Web Application
----------------------------------------------------------------
A state-of-the-art interactive digital multimedia forensics dashboard.
Designed for live lab demonstrations, viva presentations, and evidence auditing.
"""

import os
import io
import json
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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

# Page configuration
st.set_page_config(
    page_title="Image Forensic Analyzer | Digital Tampering Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-tech Cyber Security / Forensics aesthetic (Zero Red Palette)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284C7, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #90CAF9;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .verdict-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.35rem;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    /* Primary buttons: Electric Cobalt / Teal gradient - Zero Red */
    button[kind="primary"], .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    button[kind="primary"]:hover, .stButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #0F766E 100%) !important;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    /* Sliders: Blue/Cyan Accent - Zero Red */
    div[data-baseweb="slider"] div {
        background-color: #0284C7 !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #38BDF8 !important;
        border-color: #0284C7 !important;
    }
    /* Tabs: Electric Cyan Highlight */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }
    /* Radio buttons */
    div[role="radiogroup"] input:checked + div {
        border-color: #0284C7 !important;
        background-color: #0284C7 !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🔍 Image Forensic Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimedia Forensics Lab • Digital Tampering & Forgery Detection System</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Forensic Engine Settings")
st.sidebar.markdown("Tune algorithmic parameters for deep artifact inspection:")

ela_quality = st.sidebar.slider("ELA Re-save Quality (Q)", min_value=70, max_value=98, value=90, step=1,
                                help="JPEG quality factor used for re-compression error baseline.")
ela_scale = st.sidebar.slider("ELA Error Amplification Scale", min_value=5, max_value=30, value=15, step=1,
                              help="Multiplication factor to make subtle DCT errors human-visible.")
orb_features = st.sidebar.slider("ORB Max Keypoints", min_value=500, max_value=4000, value=2500, step=250,
                                 help="Max keypoints to extract for copy-move duplicate searching.")
min_spatial_dist = st.sidebar.slider("Min Spatial Distance (px)", min_value=15, max_value=80, value=35, step=5,
                                    help="Min distance between matched points to discard self-matches.")
noise_block_size = st.sidebar.slider("Noise Analysis Block Size", min_value=8, max_value=32, value=16, step=4,
                                    help="Block grid size for local noise variance mapping.")

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Details:**
- **Student**: B.Tech CSE (Cyber Security)
- **Domain**: Multimedia & Digital Forensics
- **Techniques**: ELA, ORB Copy-Move, Noise Splicing Residuals, EXIF Audit
- **Supported Formats**: JPG, JPEG, PNG, WEBP, BMP, TIFF
""")

# Input Mode Selection
mode = st.radio(
    "Select Evidence Source:",
    ["Upload New Image", "Load Verified Demonstration Dataset", "Run Benchmark Evaluation"],
    horizontal=True
)

image_to_process = None
source_name = ""

if mode == "Upload New Image":
    uploaded_file = st.file_uploader(
        "Upload Target Image (JPG, JPEG, PNG, WEBP, BMP, TIFF)",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"],
        help="Upload any original or tampered image from Google, camera, social media, or screenshot."
    )
    if uploaded_file is not None:
        image_to_process = uploaded_file
        source_name = uploaded_file.name

elif mode == "Load Verified Demonstration Dataset":
    demo_options = {
        "Authentic Landscape (Untouched)": "dataset/authentic/authentic_landscape.jpg",
        "Authentic Mountain Lake (Untouched)": "dataset/authentic/authentic_lake.jpg",
        "Copy-Move Forgery (Cloned Windmill Tower)": "dataset/tampered/copy_move/forgery_copy_move_tower.jpg",
        "Copy-Move Forgery (Cloned Sailboat)": "dataset/tampered/copy_move/forgery_copy_move_boat.jpg",
        "Splicing Forgery (Foreign Drone in Sky)": "dataset/tampered/splicing/forgery_spliced_uav.jpg",
        "Splicing Forgery (Foreign Hot-Air Balloon)": "dataset/tampered/splicing/forgery_spliced_balloon.jpg"
    }
    selected_demo = st.selectbox("Choose a pre-generated evidence sample:", list(demo_options.keys()))
    demo_path = demo_options[selected_demo]
    if os.path.exists(demo_path):
        image_to_process = demo_path
        source_name = os.path.basename(demo_path)
    else:
        st.warning("Sample dataset not found. Click below to generate it:")
        if st.button("Generate Demonstration Samples Now"):
            from generate_sample_data import generate_demonstration_dataset
            generate_demonstration_dataset()
            st.rerun()

elif mode == "Run Benchmark Evaluation":
    st.markdown("### 📊 Automated Forensics Benchmark Suite")
    st.write("Evaluates the forensic analyzer against ground truth labels across the small synthetic demonstration dataset.")
    if st.button("Run Full Benchmark & Generate Confusion Matrix", type="primary"):
        with st.spinner("Analyzing dataset images and computing metrics..."):
            from evaluate import evaluate_dataset
            metrics = evaluate_dataset()
            st.success("Benchmark completed successfully!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            c2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
            c3.metric("Recall (Sensitivity)", f"{metrics['recall'] * 100:.1f}%")
            c4.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%")

            if os.path.exists("reports/confusion_matrix.png"):
                st.image("reports/confusion_matrix.png", caption="Evaluation Confusion Matrix", width=550)

# Process Image if available
if image_to_process is not None:
    st.markdown("---")

    # Step 1: Pre-analysis Image Validation
    is_valid, err_msg, pil_img, np_img = validate_image(image_to_process)

    if not is_valid:
        st.warning(f"⚠️ Image Validation Notice: {err_msg}")
        st.info("💡 Supported formats: JPG, JPEG, PNG, WEBP, BMP, TIFF. Images must be at least 16x16 pixels and not corrupted.")
    else:
        # Step 2: Unified Forensic Processing Pipeline
        try:
            with st.spinner(f"Auditing digital image: {source_name}..."):
                # Statistical metrics
                stats = compute_image_statistics(np_img)

                # 1. EXIF and Header Metadata Audit
                meta = extract_metadata(pil_img)

                # 2. Error Level Analysis (ELA)
                ela_res = perform_ela(np_img, quality=ela_quality, scale_factor=ela_scale)

                # 3. Copy-Move Forgery Detection (ORB Feature Matching)
                cm_res = detect_copy_move(np_img, max_features=orb_features, min_distance=min_spatial_dist)

                # 4. Splicing / Local Noise Inconsistency Detection
                splice_res = detect_splicing_noise(np_img, block_size=noise_block_size)

                # 5. Multi-Modal Decision Fusion
                verdict = calculate_overall_verdict(
                    ela_score=ela_res["ela_tampering_score"],
                    copy_move_matches=cm_res["match_count"],
                    noise_score=splice_res["noise_inconsistency_score"],
                    exif_suspicious=meta["is_suspicious"]
                )

            # Final Verdict Banner (Zero red colors - high-tech indigo, amber, emerald palette)
            if verdict["risk_level"] == "HIGH":
                banner_bg = "linear-gradient(135deg, #1E1B4B 0%, #312E81 100%)"
                banner_border = "2px solid #818CF8"
                banner_icon = "⚡"
            elif verdict["risk_level"] == "MODERATE":
                banner_bg = "linear-gradient(135deg, #1E293B 0%, #334155 100%)"
                banner_border = "2px solid #F59E0B"
                banner_icon = "⚠️"
            else:
                banner_bg = "linear-gradient(135deg, #064E3B 0%, #065F46 100%)"
                banner_border = "2px solid #10B981"
                banner_icon = "✅"

            st.markdown(f"""
            <div class="verdict-box" style="background: {banner_bg}; border: {banner_border}; color: white;">
                {banner_icon} VERDICT: {verdict['verdict']}<br>
                <span style="font-size: 1.05rem; font-weight: normal; color: #E2E8F0;">
                    Forensic Risk Score: <b style="color: #67E8F9;">{verdict['forensic_risk_score']}%</b> &nbsp;|&nbsp;
                    Risk: <b>{verdict['risk_level']}</b> &nbsp;|&nbsp;
                    Primary Technique: <b>{verdict['primary_technique']}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Forensic Metric Summary Cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Forensic Risk Score", f"{verdict['forensic_risk_score']}%", delta=f"{verdict['risk_level']} Risk", delta_color="off")
            m2.metric("Copy-Move Pairs", f"{cm_res['match_count']}", delta="Cloned Keypoints" if cm_res['is_tampered'] else "Clean", delta_color="off")
            m3.metric("ELA Anomaly Score", f"{ela_res['ela_tampering_score']}/100", delta=f"{ela_res['high_error_ratio']}% Outliers", delta_color="off")
            m4.metric("Noise Discrepancy", f"{splice_res['variance_ratio']}x", delta="Potential Inconsistency" if splice_res['is_splicing_detected'] else "Uniform", delta_color="off")
            m5.metric("EXIF Status", "Suspicious" if meta["is_suspicious"] else "Clean", delta=meta.get("software_detected") or "No Editor Tag", delta_color="off")

            # Detailed Forensic Investigation Tabs
            tab_overview, tab_ela, tab_cm, tab_splice, tab_exif, tab_stats = st.tabs([
                "🖼️ Multi-Panel Overview",
                "🔬 Error Level Analysis (ELA)",
                "🎯 Copy-Move Detection",
                "🌊 Splicing & Noise Residual",
                "📋 EXIF & Header Audit",
                "📈 Image Statistical Profile"
            ])

            with tab_overview:
                st.markdown("#### Quad-View Forensic Evidence Sheet")
                r1_c1, r1_c2 = st.columns(2)
                r2_c1, r2_c2 = st.columns(2)

                with r1_c1:
                    st.image(np_img, caption=f"1. Original Target Image ({stats['width']}x{stats['height']} px, {meta.get('file_format', 'Image')})", use_container_width=True)
                with r1_c2:
                    st.image(ela_res['ela_image_np'], caption=f"2. ELA Compression Residual (Scale: {ela_scale}x)", use_container_width=True)
                with r2_c1:
                    st.image(cm_res['annotated_image'], caption=f"3. Copy-Move Cloned Points ({cm_res['match_count']} pairs)", use_container_width=True)
                with r2_c2:
                    st.image(splice_res['variance_heatmap_rgb'], caption=f"4. Potential Noise Inconsistency Map (Ratio: {splice_res['variance_ratio']}x)", use_container_width=True)

            with tab_ela:
                st.markdown("#### Error Level Analysis (ELA) Deep-Dive")
                st.write("""
                **Forensic Principle**: When an image is modified and re-saved, previously modified areas
                may experience different quantization error compared to untouched areas.
                Regions with noticeable brightness variations or distinct error patterns may serve as possible
                indicators of editing or suspicious patterns, rather than definitive proof of tampering.
                """)
                c_ela1, c_ela2 = st.columns(2)
                c_ela1.image(ela_res['ela_image_np'], caption="Enhanced Difference Map", use_container_width=True)
                c_ela2.image(ela_res['heatmap_rgb'], caption="False-Color Hotspot Heatmap (Jet Colormap)", use_container_width=True)

                st.markdown(f"""
                - **Mean Compression Error**: `{ela_res['mean_error']}`
                - **Error Standard Deviation**: `{ela_res['std_error']}`
                - **High-Error Pixel Ratio**: `{ela_res['high_error_ratio']}%`
                - **Algorithm Assessment**: {'Possible localized compression discrepancy / suspicious pattern' if ela_res['ela_tampering_score'] >= 30 else 'Uniform compression pattern across image'}
                """)

            with tab_cm:
                st.markdown("#### Copy-Move Forgery Detection (ORB Feature Matching)")
                st.write("""
                **Forensic Principle**: When objects are duplicated to hide or clone items,
                ORB features find identical textures separated by a spatial distance.
                Connecting lines highlight pairs sharing identical visual descriptors.
                """)
                st.image(cm_res['annotated_image'], caption=f"ORB Keypoints Matched: {cm_res['match_count']} pairs in {cm_res['clusters']} clusters", use_container_width=True)
                st.markdown(f"""
                - **Total Features Extracted**: `{cm_res['keypoints_found']}`
                - **Verified Duplicate Pairs**: `{cm_res['match_count']}`
                - **Spatial Translation Clusters**: `{cm_res['clusters']}`
                - **Copy-Move Confidence**: `{cm_res['confidence']}%`
                """)

            with tab_splice:
                st.markdown("#### Splicing & Noise Residual Analysis")
                st.write("""
                **Forensic Principle**: Camera sensors produce characteristic high-frequency noise distributions.
                Foreign object insertions or spliced elements may introduce localized noise variance discrepancies.
                """)
                s_c1, s_c2 = st.columns(2)
                s_c1.image(splice_res['noise_residual_rgb'], caption="Extracted High-Frequency Noise Residual", use_container_width=True)
                s_c2.image(splice_res['laplacian_residual_rgb'], caption="High-Pass Laplacian Edge Discontinuity", use_container_width=True)

                st.markdown(f"""
                - **P95 / Median Noise Variance Ratio**: `{splice_res['variance_ratio']}x`
                - **Noise Inconsistency Score**: `{splice_res['noise_inconsistency_score']} / 100`
                - **Noise Consistency Assessment**: {'Potential noise inconsistency detected' if splice_res['is_splicing_detected'] else 'Relatively uniform noise distribution'}
                """)

            with tab_exif:
                st.markdown("#### EXIF Metadata & Digital Footprint Audit")
                if meta["exif_found"] or meta.get("metadata_found"):
                    st.success(f"Metadata Headers Found in Image File (Format: {meta.get('file_format', 'Unknown')}).")
                    if meta["is_suspicious"]:
                        st.warning("⚠️ Suspicious Indicators:")
                        for r in meta["reasons"]:
                            st.write(f"- {r}")
                    else:
                        st.info("No editing software signatures found.")

                    c_cam, c_soft = st.columns(2)
                    c_cam.write(f"**Camera Hardware:** `{meta.get('camera_model') or 'Not Specified'}`")
                    c_soft.write(f"**Software Used:** `{meta.get('software_detected') or 'None detected'}`")

                    if meta.get("tags"):
                        with st.expander("View Full Raw Metadata Tags Table"):
                            st.json(meta["tags"])
                else:
                    st.info("No EXIF metadata found. Most social media platforms (WhatsApp, Facebook, Twitter, Instagram) and screenshots strip metadata on creation/upload.")

            with tab_stats:
                st.markdown("#### Mathematical & Statistical Properties")
                st.json(stats)

            # Export Report Button
            st.markdown("---")
            report_json_str = json.dumps({
                "source_evidence": source_name,
                "final_verdict": verdict,
                "image_stats": stats,
                "ela_analysis": {
                    "score": ela_res["ela_tampering_score"],
                    "mean_error": ela_res["mean_error"],
                    "std_error": ela_res["std_error"],
                    "high_error_ratio": ela_res["high_error_ratio"],
                    "interpretation": "Possible localized compression discrepancy" if ela_res["ela_tampering_score"] >= 30 else "Uniform compression pattern"
                },
                "copy_move_analysis": {
                    "matched_pairs": cm_res["match_count"],
                    "clusters": cm_res["clusters"],
                    "module_confidence": cm_res["confidence"],
                    "is_tampered": cm_res["is_tampered"]
                },
                "noise_analysis": {
                    "noise_inconsistency_score": splice_res["noise_inconsistency_score"],
                    "variance_ratio": splice_res["variance_ratio"],
                    "assessment": "Potential noise inconsistency detected" if splice_res["is_splicing_detected"] else "Relatively uniform noise distribution"
                },
                "metadata_analysis": {
                    "exif_found": meta["exif_found"],
                    "is_suspicious": meta["is_suspicious"],
                    "software_detected": meta.get("software_detected"),
                    "camera_model": meta.get("camera_model"),
                    "audit_notes": meta["reasons"]
                },
                "ela_summary": ela_res["ela_tampering_score"],
                "copy_move_matches": cm_res["match_count"],
                "splicing_score": splice_res["noise_inconsistency_score"],
                "exif_summary": meta["reasons"]
            }, indent=4, default=str)

            st.download_button(
                label="📥 Download Official Forensic Report (JSON)",
                data=report_json_str,
                file_name=f"Forensic_Report_{os.path.splitext(source_name)[0]}.json",
                mime="application/json"
            )

        except Exception as e:
            st.warning(f"⚠️ Forensic analysis notice while processing '{source_name}': {str(e)}")
            st.info("Please verify that the uploaded image is a valid, readable graphic file.")
