import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class MoveNetPoseAnalyzer:
    def __init__(self, model_name="lightning"):
        model_url = {
            "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
            "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        }[model_name]

        self.model = hub.load(model_url)
        self.input_size = 256

    def analyze(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize_with_pad(np.expand_dims(img_rgb, axis=0), self.input_size, self.input_size)
        input_tensor = tf.cast(resized, dtype=tf.int32)
        outputs = self.model.signatures['serving_default'](input_tensor)
        keypoints = outputs['output_0'].numpy()[0][0]
        return keypoints

    def draw_landmarks(self, frame, keypoints):
        h, w, _ = frame.shape
        for kp in keypoints:
            y, x, confidence = kp[0], kp[1], kp[2]
            if confidence > 0.3:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        return frame

    def release(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    analyzer = MoveNetPoseAnalyzer(model_name="thunder")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = analyzer.analyze(frame)
            frame = analyzer.draw_landmarks(frame, keypoints)

            cv2.imshow("MoveNet Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        analyzer.release()
