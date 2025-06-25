import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(refine_face_landmarks=True)
drawing = mp.solutions.drawing_utils

def process_frame(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    if results.pose_landmarks:
        drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

async def handler(websocket):
    async for message in websocket:
        processed = process_frame(message)
        await websocket.send(processed)

async def main():
    async with websockets.serve(handler, "localhost", 8765, max_size=2**25):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
