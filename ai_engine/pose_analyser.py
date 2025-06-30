import cv2
import mediapipe as mp
import math

class HolisticAnalyzer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.drawing_utils = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True
        )

    def analyze(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        return results

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

        if results.left_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        if results.right_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        if results.face_landmarks:
            self.drawing_utils.draw_landmarks(
                frame, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)

        return frame

    def calculate_angle(self, a, b, c):
        """Calculate angle between 3 points."""
        ax, ay = a
        bx, by = b
        cx, cy = c

        angle = math.degrees(
            math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
        )
        return abs(angle) if angle >= 0 else abs(360 + angle)

    def analyze_pose_angles(self, frame, landmarks):
        if not landmarks or not landmarks.pose_landmarks:
            return frame

        h, w, _ = frame.shape
        pose = landmarks.pose_landmarks.landmark

        def get_point(idx):
            return int(pose[idx].x * w), int(pose[idx].y * h)

        # Elbow Angles
        l_shoulder = get_point(self.mp_holistic.PoseLandmark.LEFT_SHOULDER)
        l_elbow = get_point(self.mp_holistic.PoseLandmark.LEFT_ELBOW)
        l_wrist = get_point(self.mp_holistic.PoseLandmark.LEFT_WRIST)

        r_shoulder = get_point(self.mp_holistic.PoseLandmark.RIGHT_SHOULDER)
        r_elbow = get_point(self.mp_holistic.PoseLandmark.RIGHT_ELBOW)
        r_wrist = get_point(self.mp_holistic.PoseLandmark.RIGHT_WRIST)

        nose = get_point(self.mp_holistic.PoseLandmark.NOSE)
        mid_shoulder = (
            int((l_shoulder[0] + r_shoulder[0]) / 2),
            int((l_shoulder[1] + r_shoulder[1]) / 2)
        )

        neck_angle = self.calculate_angle(l_shoulder, nose, r_shoulder)
        left_elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)

        cv2.putText(frame, f"Neck Angle: {int(neck_angle)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Left Elbow: {int(left_elbow_angle)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right Elbow: {int(right_elbow_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame


    def release(self):
        self.holistic.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.holistic.close()
        self.holistic = self.mp_holistic.Holistic()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    analyzer = HolisticAnalyzer()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = analyzer.analyze(frame)
            frame = analyzer.draw_landmarks(frame, results)
            frame = analyzer.analyze_pose_angles(frame, results)

            cv2.imshow("Pose Analysis with Angles", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        analyzer.release()

# --- Utility functions for external scripts ---

# List of pose landmarks to extract (MediaPipe names)
POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

_analyzer_instance = None

def _get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HolisticAnalyzer()
    return _analyzer_instance

def detect_pose(frame):
    """
    Detect pose landmarks and return a dict of keypoints: {name: (x, y) or None}
    """
    analyzer = _get_analyzer()
    results = analyzer.analyze(frame)
    keypoints = {}
    h, w, _ = frame.shape
    if results.pose_landmarks:
        for name, idx in POSE_LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            if lm.visibility > 0.5:
                keypoints[name] = (int(lm.x * w), int(lm.y * h))
            else:
                keypoints[name] = None
    else:
        for name in POSE_LANDMARKS:
            keypoints[name] = None
    return keypoints

def analyze_posture(keypoints, reference_keypoints=None):
    """
    Stub for posture analysis. If reference_keypoints is provided, returns average distance.
    """
    if reference_keypoints is None:
        return None
    import numpy as np
    dists = []
    for k in keypoints:
        if k in reference_keypoints and keypoints[k] is not None and reference_keypoints[k] is not None:
            dists.append(np.linalg.norm(np.array(keypoints[k]) - np.array(reference_keypoints[k])))
    return np.mean(dists) if dists else float('inf')
