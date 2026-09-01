"""
classify_image.py
=================
Image Classification and Identity Verification CLI for Classical ML Pipelines.

Replaces live webcam demo with direct single-image inference.

Usage:
  1. Classify a specific image:
     python scripts/classify_image.py --image data/raw/Aashiq/Aashiq_0001.jpg

  2. Classify a random unseen image from the test set:
     python scripts/classify_image.py --random

  3. Pass positional argument:
     python scripts/classify_image.py data/raw/unknown/unknown_0005.jpg
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import cv2
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.preprocessing.detect_align import FacePreprocessor
from src.open_set.intruder_detector import IntruderDetector
from src.classifiers.pipelines import PIPELINE_CONFIGS
from scripts.train_models import train_and_save_models


def load_or_train_models(model_path: str = "models/review1_models.joblib") -> Dict[str, Any]:
    """Load pre-trained model bundle or train if missing."""
    full_path = os.path.join(WORKSPACE_ROOT, model_path)
    if not os.path.exists(full_path):
        print(f"[Notice] No pre-trained model bundle found at: {full_path}")
        print("  -> Automatically training all 5 classical ML pipelines once...")
        train_and_save_models(model_save_path=model_path)
    
    return joblib.load(full_path)


def classify_image(
    image_path: Optional[str] = None,
    use_random_test: bool = False,
    output_vis_path: Optional[str] = "reports/figures/classification_result.jpg",
    model_path: str = "models/review1_models.joblib"
) -> Dict[str, Any]:
    print("=" * 85)
    print("           CLASSICAL ML FACE RECOGNITION & INTRUDER DETECTION CLASSIFIER")
    print("=" * 85)

    # 1. Load trained bundle
    bundle = load_or_train_models(model_path)
    pipelines = bundle["trained_pipelines"]
    known_classes = bundle["known_classes"]
    all_classes = bundle["all_classes"]

    detector = IntruderDetector(
        trained_pipelines=pipelines,
        known_classes=known_classes,
        prob_threshold=bundle.get("prob_threshold", 0.35),
        knn_dist_threshold=bundle.get("knn_dist_threshold", 80.0),
        min_consensus_votes=bundle.get("min_consensus_votes", 2),
        enrolled_templates=bundle.get("enrolled_templates", {}),
        template_thresholds=bundle.get("template_thresholds", {})
    )

    preprocessor = FacePreprocessor(target_size=(128, 128))

    raw_image = None
    ground_truth = None
    input_source_desc = ""

    # 2. Handle input image source
    if use_random_test or image_path is None:
        test_images = bundle.get("test_images", None)
        test_labels = bundle.get("test_labels", None)

        if test_images is not None and len(test_images) > 0:
            rand_idx = random.randint(0, len(test_images) - 1)
            normalized_face = test_images[rand_idx]
            ground_truth = str(test_labels[rand_idx])
            input_source_desc = f"Random Unseen Test Sample #{rand_idx} (Ground Truth: {ground_truth})"
            # Create synthetic 3-channel visual representation for saving
            raw_image = cv2.cvtColor(normalized_face, cv2.COLOR_GRAY2BGR)
            bbox = (10, 10, 108, 108)
        else:
            print("[Error] No test pool found in model bundle. Please specify --image <path>.")
            return {}
    else:
        if not os.path.exists(image_path):
            # Check relative to workspace
            alt_path = os.path.join(WORKSPACE_ROOT, image_path)
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                print(f"[Error] Image file not found: {image_path}")
                return {}

        input_source_desc = f"Image File: {image_path}"
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            # Try unicode safe load
            data = np.fromfile(image_path, dtype=np.uint8)
            raw_image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if raw_image is None:
            print(f"[Error] Could not decode image file: {image_path}")
            return {}

        # Attempt to infer ground truth from directory name
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
        if parent_dir in all_classes:
            ground_truth = parent_dir

        # Preprocess face (detect, horizontal eye align, CLAHE normalize)
        normalized_face, bbox = preprocessor.process(raw_image)

    print(f"\n[Input]: {input_source_desc}")
    if ground_truth:
        print(f"[Ground Truth Label]: {ground_truth}")

    # 3. Evaluate through all 5 Classical ML Pipelines + Consensus Fusion
    fusion_result = detector.predict_fusion(normalized_face)

    # 4. Display Formatted Results Table
    print("\n" + "-" * 85)
    print(f"{'Pipeline':<35} | {'Extractor':<16} | {'Classifier':<16} | {'Prediction':<12} | {'Confidence'}")
    print("-" * 85)

    for p_id in range(1, 6):
        res = fusion_result["individual_results"][p_id]
        p_name = res["pipeline_name"]
        f_name = res["feature_name"]
        c_name = res["classifier_name"]
        pred = res["final_decision"]
        conf = f"{res['confidence']*100:.1f}%"
        print(f"{p_name:<35} | {f_name:<16} | {c_name:<16} | {pred:<12} | {conf}")

    print("-" * 85)

    # 5. Display Final Unified Decision Banner
    is_auth = fusion_result["is_authorized"]
    top_id = fusion_result["identity"]
    status_text = f"AUTHORIZED: {top_id}" if is_auth else "ALERT: INTRUDER / UNKNOWN DETECTED"
    status_color_box = "\033[92m" if is_auth else "\033[91m"
    reset_color = "\033[0m"

    print(f"\n{'=' * 85}")
    print(f"  FINAL CLASSIFICATION: {status_color_box}{status_text}{reset_color}")
    print(f"  Mean Model Confidence:  {fusion_result['mean_confidence']*100:.2f}%")
    print(f"  Model Consensus Votes:  {fusion_result['consensus_votes']}")
    if ground_truth:
        match_status = "CORRECT MATCH" if (is_auth and top_id == ground_truth) or (not is_auth and ground_truth.lower() in ['unknown', 'intruder']) else "MISMATCH"
        print(f"  Ground Truth Verification: {match_status} (Expected: {ground_truth})")
    print(f"{'=' * 85}\n")

    # 6. Save visual annotated output
    if output_vis_path and raw_image is not None:
        out_full = os.path.join(WORKSPACE_ROOT, output_vis_path)
        os.makedirs(os.path.dirname(out_full), exist_ok=True)

        vis_img = raw_image.copy()
        H, W = vis_img.shape[:2]

        if bbox is not None:
            bx, by, bw, bh = bbox
            box_col = (0, 200, 0) if is_auth else (0, 0, 220)
            cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), box_col, 2)

            # Overlay banner
            cv2.rectangle(vis_img, (0, 0), (W, 45), (30, 30, 30), -1)
            cv2.putText(vis_img, status_text, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, box_col, 2)

        cv2.imwrite(out_full, vis_img)
        print(f"  -> Saved visual classification result to: {out_full}")

    return {
        "status": fusion_result["status"],
        "identity": fusion_result["identity"],
        "is_authorized": fusion_result["is_authorized"],
        "mean_confidence": fusion_result["mean_confidence"],
        "consensus_votes": fusion_result["consensus_votes"],
        "individual_results": fusion_result["individual_results"],
        "ground_truth": ground_truth
    }


def main():
    parser = argparse.ArgumentParser(description="Classify an input face image across 5 Classical ML pipelines.")
    parser.add_argument("input_image", nargs="?", default=None, help="Path to input face image file (optional positional)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input face image file")
    parser.add_argument("--random", "-r", action="store_true", help="Classify a random unseen image from the test set")
    parser.add_argument("--output", "-o", type=str, default="reports/figures/classification_result.jpg", help="Path to save visual output result")
    parser.add_argument("--model-path", type=str, default="models/review1_models.joblib", help="Path to saved joblib model bundle")

    args = parser.parse_args()

    chosen_image = args.image or args.input_image

    if not chosen_image and not args.random:
        # Default to random unseen test image
        args.random = True

    classify_image(
        image_path=chosen_image,
        use_random_test=args.random,
        output_vis_path=args.output,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()
