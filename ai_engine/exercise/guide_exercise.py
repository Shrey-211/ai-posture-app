"""
guide_exercise.py

Guides the user through a saved reference exercise, overlaying reference keypoints and advancing when the user's posture aligns.
"""

import cv2
import json
import os
import sys
import glob
import numpy as np
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ai_engine import pose_analyser

KEY_LANDMARKS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']

MAJOR_KEYPOINTS = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee'
]

def is_fully_in_frame(keypoints):
    """Check if all key landmarks are detected (not None)."""
    if not keypoints:
        return False
    for lm in KEY_LANDMARKS:
        if lm not in keypoints or keypoints[lm] is None:
            return False
    return True

def select_reference_file():
    """Prompt the user to select a reference exercise JSON file from the project root."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    files = glob.glob(os.path.join(project_root, "reference_exercise_*.json"))
    if not files:
        print("No reference exercise files found.")
        return None
    print("Available reference exercises:")
    for idx, fname in enumerate(files):
        print(f"{idx+1}: {os.path.basename(fname)}")
    while True:
        try:
            choice = int(input(f"Select an exercise (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                return files[choice-1]
        except Exception:
            pass
        print("Invalid selection. Please try again.")

def get_torso_length(keypoints):
    """Compute torso length as the distance between mid-shoulder and mid-hip."""
    try:
        ls, rs = keypoints['left_shoulder'], keypoints['right_shoulder']
        lh, rh = keypoints['left_hip'], keypoints['right_hip']
        mid_shoulder = np.mean([ls, rs], axis=0)
        mid_hip = np.mean([lh, rh], axis=0)
        return np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip))
    except Exception:
        return None

def adjust_reference_to_user(ref_kps, user_kps):
    """
    Scale and translate reference keypoints to match user's body proportions.
    Uses torso length and mid-shoulder/mid-hip as anchors.
    """
    # Compute reference and user torso lengths and centers
    ref_torso = get_torso_length(ref_kps)
    user_torso = get_torso_length(user_kps)
    if not ref_torso or not user_torso:
        return ref_kps  # fallback: no adjustment

    # Compute centers
    ref_center = np.mean([ref_kps['left_hip'], ref_kps['right_hip'],
                          ref_kps['left_shoulder'], ref_kps['right_shoulder']], axis=0)
    user_center = np.mean([user_kps['left_hip'], user_kps['right_hip'],
                           user_kps['left_shoulder'], user_kps['right_shoulder']], axis=0)
    scale = user_torso / ref_torso
    adjusted = {}
    for name, coord in ref_kps.items():
        if coord is not None:
            vec = np.array(coord) - ref_center
            adj_coord = user_center + vec * scale
            adjusted[name] = tuple(adj_coord)
        else:
            adjusted[name] = None
    return adjusted

def average_keypoint_distance(kps1, kps2):
    """Compute average Euclidean distance between corresponding keypoints."""
    dists = []
    for k in kps1:
        if k in kps2 and kps1[k] is not None and kps2[k] is not None:
            dists.append(np.linalg.norm(np.array(kps1[k]) - np.array(kps2[k])))
    return np.mean(dists) if dists else float('inf')

def keypoints_within_threshold(kps1, kps2, threshold, keypoints_list=None):
    """Return the fraction of keypoints within the threshold distance, for a given list."""
    close = 0
    total = 0
    keys = keypoints_list if keypoints_list is not None else kps1.keys()
    for k in keys:
        if k in kps1 and k in kps2 and kps1[k] is not None and kps2[k] is not None:
            dist = np.linalg.norm(np.array(kps1[k]) - np.array(kps2[k]))
            if dist < threshold:
                close += 1
            total += 1
    return close / total if total > 0 else 0

def main():
    ref_file = select_reference_file()
    if not ref_file:
        return

    try:
        with open(ref_file, "r") as f:
            ref_data = json.load(f)
    except Exception as e:
        print(f"Error loading reference file: {e}")
        return

    ref_sequence = ref_data.get("keypoints_sequence", [])
    if not ref_sequence:
        print("Reference file contains no keypoints.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Align your body in the frame. Exercise will start when you are detected.")
    user_ready = False
    user_first_kps = None

    # Wait for user to be fully in frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera.")
            break
        keypoints = pose_analyser.detect_pose(frame)
        if keypoints:
            for name, coord in keypoints.items():
                if coord is not None:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)
        if is_fully_in_frame(keypoints):
            user_ready = True
            user_first_kps = keypoints
            cv2.putText(frame, "Ready! Starting exercise...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Guide Exercise", frame)
            cv2.waitKey(1000)
            break
        else:
            cv2.putText(frame, "Please ensure your full body is in the frame", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Guide Exercise", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    pose_idx = 0
    completed = False
    torso_length = get_torso_length(user_first_kps)
    if torso_length is None:
        print("Warning: Could not compute torso length. Using default threshold of 30 pixels.")
        threshold = 30.0
    else:
        threshold = 0.08 * torso_length  # 8% of torso length

    while not completed:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera.")
            break
        keypoints = pose_analyser.detect_pose(frame)
        # Overlay user landmarks
        if keypoints:
            for name, coord in keypoints.items():
                if coord is not None:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)
        # Dynamically adjust reference pose to user's current body
        if keypoints and is_fully_in_frame(keypoints):
            ref_kps = adjust_reference_to_user(ref_sequence[pose_idx], keypoints)
        else:
            ref_kps = {k: None for k in ref_sequence[pose_idx]}
        # Overlay reference landmarks (red)
        for name, coord in ref_kps.items():
            if coord is not None:
                cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 0, 255), -1)

        # --- Feedback window ---
        feedback_img = 255 * np.ones((400, 400, 3), dtype=np.uint8)
        y0, dy = 30, 28
        cv2.putText(feedback_img, f"Pose Feedback ({pose_idx+1}/{len(ref_sequence)})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        line = 1
        fraction_close = 0.0
        if is_fully_in_frame(keypoints):
            # Calculate fraction for MAJOR_KEYPOINTS
            fraction_close = keypoints_within_threshold(keypoints, ref_kps, threshold, MAJOR_KEYPOINTS)
            for k in MAJOR_KEYPOINTS:
                user_pt = keypoints.get(k)
                ref_pt = ref_kps.get(k)
                if user_pt is not None and ref_pt is not None:
                    dx = ref_pt[0] - user_pt[0]
                    dy_ = ref_pt[1] - user_pt[1]
                    dist = np.linalg.norm(np.array(user_pt) - np.array(ref_pt))
                    direction = []
                    if abs(dx) > 10:
                        direction.append('right' if dx > 0 else 'left')
                    if abs(dy_) > 10:
                        direction.append('down' if dy_ > 0 else 'up')
                    if dist > threshold:
                        msg = f"{k}: Move {', '.join(direction)} ({int(dist)} px)"
                        color = (0, 0, 255)
                    else:
                        msg = f"{k}: OK"
                        color = (0, 128, 0)
                    cv2.putText(feedback_img, msg, (10, y0 + line * dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    line += 1
            # Show fraction in feedback
            cv2.putText(feedback_img, f"Aligned: {int(fraction_close*100)}%", (10, y0 + line * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.putText(feedback_img, "Please ensure your full body is in the frame", (10, y0 + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Guide Exercise", frame)
        cv2.imshow("Pose Feedback", feedback_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

        # Compare user pose to reference
        if is_fully_in_frame(keypoints):
            # Use the same fraction as in feedback
            if fraction_close >= 0.8 - 1e-6:
                cv2.putText(frame, "Aligned!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                pose_idx += 1
                if pose_idx >= len(ref_sequence):
                    completed = True
            else:
                cv2.putText(frame, f"Align to reference pose ({pose_idx+1}/{len(ref_sequence)})", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if completed:
        print("Exercise complete!")
        # Show completion message
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "Exercise Complete!", (60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Guide Exercise", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()