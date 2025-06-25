import cv2
import mediapipe as mp

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

    def analyze(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        if landmarks:
            self.drawing_utils.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def release(self):
        self.pose.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.pose.close()
        self.pose = self.mp_pose.Pose()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pose_analyzer = PoseAnalyzer()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = pose_analyzer.analyze(frame)
            frame = pose_analyzer.draw_landmarks(frame, landmarks)

            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        pose_analyzer.release()
