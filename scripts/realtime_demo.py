"""
realtime_demo.py
================
Real-Time Live Camera Demonstration of Intruder Detection.

Displays:
- Real-time video stream with detected face bounding box and alignment info.
- Side-by-side predictions from all 5 Classical Pipelines:
    1. Pipeline 1: BSIF + Random Forest
    2. Pipeline 2: LPQ + k-NN
    3. Pipeline 3: WLD + Logistic Regression
    4. Pipeline 4: Gabor + RBF-SVM
    5. Pipeline 5: Landmark Geometry + Decision Tree
- Real-time Open-Set Intruder Rejection Banner:
    [ GREEN: AUTHORIZED - <Person Name> ] or [ RED: INTRUDER / UNKNOWN ALERT ]
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import joblib
from src.preprocessing.detect_align import FacePreprocessor
from src.open_set.intruder_detector import IntruderDetector
from scripts.train_models import train_and_save_models


def run_live_demo(camera_id: int = 0) -> None:
    print("=" * 80)
    print("  CLASSICAL CV & ML: INTRUDER DETECTION & MULTI-PIPELINE DEMO")
    print("=" * 80)

    preprocessor = FacePreprocessor(target_size=(128, 128))
    model_path = os.path.join(WORKSPACE_ROOT, "models", "review1_models.joblib")

    # Check if pre-trained models exist; if not, train and save once
    if not os.path.exists(model_path):
        print(f"\n[Notice] No pre-trained model bundle found at: {model_path}")
        print("  -> Training models once and saving to disk...")
        train_and_save_models(model_save_path="models/review1_models.joblib")

    # Load pre-trained models instantly in milliseconds
    print(f"\n[1/2] Loading pre-trained 5-Pipeline bundle from: {model_path}...")
    bundle = joblib.load(model_path)
    detector = IntruderDetector(
        trained_pipelines=bundle["trained_pipelines"],
        known_classes=bundle["known_classes"],
        prob_threshold=bundle.get("prob_threshold", 0.35),
        knn_dist_threshold=bundle.get("knn_dist_threshold", 80.0),
        min_consensus_votes=bundle.get("min_consensus_votes", 2),
        enrolled_templates=bundle.get("enrolled_templates", {}),
        template_thresholds=bundle.get("template_thresholds", {})
    )
    print(f"  -> Enrolled Authorized Classes: {bundle['known_classes']}")

    print("\n[2/2] Initializing Camera Stream...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open video capture device {camera_id}.")
        return

    print("\n[3/3] Live Demonstration Running! Press 'q' to exit.")

    fps_list = []

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        display_frame = frame.copy()

        # Detect and align face
        normalized_face, bbox = preprocessor.process(frame)

        if bbox is not None:
            bx, by, bw, bh = bbox

            # Run through Intruder Detector Multi-Pipeline Fusion
            fusion_res = detector.predict_fusion(normalized_face)

            is_auth = fusion_res["is_authorized"]
            top_id = fusion_res["identity"]
            confidence = fusion_res["mean_confidence"]

            # Visual alert colors: Green for Authorized, Red for Intruder
            box_color = (0, 220, 0) if is_auth else (0, 0, 255)
            banner_text = f"AUTHORIZED: {top_id}" if is_auth else "ALERT: INTRUDER DETECTED"

            # Draw face bounding box
            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), box_color, 3)

            # Top Header Alert Banner
            cv2.rectangle(display_frame, (0, 0), (W, 55), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                banner_text,
                (20, 38),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                box_color,
                2
            )

            # Draw Side Dashboard displaying individual member outputs
            dashboard_w = 320
            cv2.rectangle(display_frame, (W - dashboard_w, 60), (W - 10, 260), (30, 30, 30), -1)
            cv2.rectangle(display_frame, (W - dashboard_w, 60), (W - 10, 260), (80, 80, 80), 1)

            cv2.putText(display_frame, "MEMBER PIPELINES", (W - dashboard_w + 10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            y_offset = 110
            for m_id in range(1, 6):
                m_res = fusion_res["individual_results"][m_id]
                m_txt = f"M{m_id} ({m_res['feature_name']}): {m_res['final_decision']}"
                col = (0, 255, 0) if m_res['final_decision'] != "INTRUDER" else (100, 100, 255)
                cv2.putText(display_frame, m_txt, (W - dashboard_w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
                y_offset += 26

        # Calculate FPS
        t_elapsed = time.time() - t_start
        fps = 1.0 / max(t_elapsed, 1e-4)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        # Display frame
        window_name = "Review 1: Real-Time Intruder Detection"
        cv2.imshow(window_name, display_frame)

        # Check for exit key ('q', 'Q', ESC) or window close button
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # 27 = ESC key
            print("\n[Demo] Exit key pressed. Terminating live stream...")
            break

        # Check if the user closed the window with 'X'
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\n[Demo] Window closed by user. Terminating live stream...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()
