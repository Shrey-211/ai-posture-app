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

def keypoints_within_threshold(kps1, kps2, threshold):
    """Return the fraction of keypoints within the threshold distance."""
    close = 0
    total = 0
    for k in kps1:
        if k in kps2 and kps1[k] is not None and kps2[k] is not None:
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

        # Compare user pose to reference
        if is_fully_in_frame(keypoints):
            fraction_close = keypoints_within_threshold(keypoints, ref_kps, threshold)
            if fraction_close >= 0.8:
                cv2.putText(frame, "Aligned!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                pose_idx += 1
                if pose_idx >= len(ref_sequence):
                    completed = True
            else:
                cv2.putText(frame, f"Align to reference pose ({pose_idx+1}/{len(ref_sequence)})", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Please ensure your full body is in the frame", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Guide Exercise", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

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