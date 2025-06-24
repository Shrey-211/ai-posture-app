from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
from ai_engine.pose_analyser import PoseAnalyzer

app = FastAPI()
pose_analyzer = PoseAnalyzer()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            print("Received frame data:", data[:30], "...")  # preview start of base64 string

            image_data = base64.b64decode(data)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode image.")
                continue

            landmarks = pose_analyzer.analyze(frame)
            await websocket.send_json({"landmarks": str(landmarks is not None)})
        except Exception as e:
            print("Error:", e)
            break
