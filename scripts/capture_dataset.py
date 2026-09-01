"""
capture_dataset.py
==================
Interactive Face Dataset Collection Script for Enrolled Members & Unknown Intruders.

Features:
- Live camera stream with face detection bounding box and eye-alignment overlay.
- Automatic capture intervals (e.g. 1 frame every 0.3s).
- Direct storage in structured directories: `data/raw/<person_name>/`.
- Guides user to vary facial poses (straight, left, right, up, down) and expressions.
"""

import os
import sys
import time
from pathlib import Path
import cv2

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.preprocessing.detect_align import FacePreprocessor


def capture_faces_for_subject(
    person_name: str,
    target_count: int = 80,
    camera_id: int = 0,
    delay_between_frames_sec: float = 0.3
) -> None:
    """
    Capture live face frames from webcam for a given person identity.

    Args:
        person_name: Identity folder name (e.g. 'Aashiq', 'Member2', 'unknown').
        target_count: Target number of images to capture (default: 80).
        camera_id: Video capture device index (default: 0).
        delay_between_frames_sec: Delay between automatic captures.
    """
    output_dir = os.path.join(WORKSPACE_ROOT, "data", "raw", person_name)
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = FacePreprocessor(target_size=(128, 128))
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[Error] Could not open video device index {camera_id}.")
        return

    print("=" * 70)
    print(f"  FACE DATASET CAPTURE TOOL: Enrolling '{person_name}'")
    print(f"  Target: {target_count} images -> {output_dir}")
    print("  Controls: [Space] = Start/Pause Auto Capture | [Q] = Quit")
    print("=" * 70)

    import glob
    existing_files = glob.glob(os.path.join(output_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(output_dir, "*.[pP][nN][gG]"))
    start_index = len(existing_files)

    captured_this_session = 0
    auto_capture = False
    last_capture_time = 0.0

    print(f"  [Info] Existing images in '{person_name}': {start_index}. Appending new captures...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab camera frame.")
            break

        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # Detect face
        bbox = preprocessor.detect_face(frame)

        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                "Face Detected",
                (bx, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Check if auto capture is active and delay elapsed
            now = time.time()
            if auto_capture and (now - last_capture_time >= delay_between_frames_sec):
                # Save raw frame with non-colliding sequential index
                current_idx = start_index + captured_this_session + 1
                filename = f"{person_name}_{current_idx:04d}.jpg"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, frame)
                captured_this_session += 1
                last_capture_time = now
                print(f"  [Captured {captured_this_session}/{target_count}] Saved {filename} (Total in folder: {start_index + captured_this_session})")

                if captured_this_session >= target_count:
                    print(f"\n[Success] Reached session target of {target_count} images for '{person_name}'!")
                    break

        # UI Overlay
        total_in_dir = start_index + captured_this_session
        status_text = f"Subject: {person_name} | Session: {captured_this_session}/{target_count} | Total: {total_in_dir}"
        mode_text = "AUTO CAPTURING..." if auto_capture else "PAUSED (Press SPACE to start)"
        cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display_frame, mode_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if auto_capture else (0, 165, 255), 2)

        window_title = "Face Dataset Collector"
        cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\n[Terminated] User exited collection.")
            break
        elif key == 32:  # Space bar
            auto_capture = not auto_capture

        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            print("\n[Terminated] Window closed by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
    else:
        target_name = input("Enter Person Name (or 'unknown'): ").strip() or "Subject1"

    capture_faces_for_subject(person_name=target_name, target_count=80)
