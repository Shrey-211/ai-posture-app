"""
record_exercise.py

Records a reference exercise using webcam video, overlays pose landmarks, and saves the sequence of keypoints to a JSON file.
"""

import cv2
import json
import os
import sys
import time
from datetime import datetime
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ai_engine import pose_analyser

# Key landmarks to check for full-body detection
KEY_LANDMARKS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']

def is_fully_in_frame(keypoints):
    """Check if all key landmarks are detected (not None)."""
    if not keypoints:
        return False
    for lm in KEY_LANDMARKS:
        if lm not in keypoints or keypoints[lm] is None:
            return False
    return True

def prompt_exercise_name():
    """Prompt the user for an exercise name."""
    name = input("Enter a name for this exercise: ").strip()
    while not name:
        name = input("Exercise name cannot be empty. Please enter a name: ").strip()
    return name

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press SPACE to record a pose. Press ESC to finish and save.")
    recorded_keypoints = []
    fully_in_frame = False
    pose_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera.")
            break

        # Detect pose landmarks
        keypoints = pose_analyser.detect_pose(frame)
        # Overlay landmarks on frame
        if keypoints:
            for name, coord in keypoints.items():
                if coord is not None:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)

        # Check if user is fully in frame
        fully_in_frame = is_fully_in_frame(keypoints)
        if not fully_in_frame:
            cv2.putText(frame, "Please ensure your full body is in the frame", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display number of recorded poses
        cv2.putText(frame, f"Recorded poses: {pose_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Record Exercise", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to exit and save
            break
        elif key == 32:  # SPACE to record current pose
            if fully_in_frame:
                recorded_keypoints.append({k: v for k, v in keypoints.items()})
                pose_count += 1
                print(f"Pose {pose_count} recorded!")
            else:
                print("Cannot record: Please ensure your full body is in the frame.")

    if recorded_keypoints:
        exercise_name = prompt_exercise_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "exercise_name": exercise_name,
            "timestamp": timestamp,
            "keypoints_sequence": recorded_keypoints
        }
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        filename = f"reference_exercise_{exercise_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Exercise saved to {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No poses recorded. Nothing was saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()