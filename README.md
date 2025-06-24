
---

## 🧠 AI Exercise Posture Detection — Local Setup (FastAPI + MediaPipe + WebSocket)

This app uses your webcam to capture video, sends each frame to a FastAPI backend using WebSocket, and analyzes posture in real-time using **MediaPipe Pose**.

---

## 📁 Project Structure

```
ai-exercise-app/
├── main.py                      # FastAPI backend
├── requirements.txt             # Python dependencies
├── ai_engine/
│   ├── __init__.py
│   └── pose_analyzer.py         # Pose detection logic using MediaPipe
├── static/
│   └── index.html               # WebSocket + webcam test client
└── uploads/ (optional)          # Store frames or logs if needed
```

---

## 🛠️ Requirements

* Python 3.10 or 3.11 (❗**Do NOT use Python 3.13** — `mediapipe` doesn't support it yet)
* pip

---

## ✅ Setup Instructions

### 🔹 Step 1: Clone or Download the Project

```bash
cd path/to/your/projects
mkdir ai-exercise-app && cd ai-exercise-app
# Add files: main.py, requirements.txt, ai_engine/ and static/
```

---

### 🔹 Step 2: Create and Activate a Virtual Environment

```bash
# Windows
py -3.10 -m venv ai_posture_env
.\ai_posture_env\Scripts\activate
```

---

### 🔹 Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you're missing the file, create `requirements.txt`:

```txt
fastapi
uvicorn
mediapipe
opencv-python
websockets
aiofiles
python-multipart
```

---

### 🔹 Step 4: Start the Backend Server

```bash
uvicorn main:app --reload
```

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### 🔹 Step 5: Open the Frontend WebCam Client

In your browser, open:

```
http://localhost:8000/static/index.html
```

* Allow camera access
* It will capture a frame every 300ms, send it to backend, and receive landmark data

---

## 📡 How It Works

1. `index.html` accesses your webcam and sends frames via WebSocket
2. `main.py` receives base64 frames, decodes them, and uses `pose_analyzer.py` for pose detection
3. FastAPI sends back whether pose landmarks were detected

---

## 🧪 Debug Tips

* If the WebSocket closes immediately, check console logs in FastAPI
* Add `print()` in your WebSocket handler to inspect incoming data
* Make sure image data is base64-only (`data:image/jpeg;base64,...` → strip header)

---

## 🚀 Next Steps

* Display landmarks on frontend (using Canvas)
* Integrate pose score or feedback (e.g., “raise arm higher”)
* Stream from Android camera to backend (instead of browser)

---

## 📞 Contact

Built by Shreyas Jadhav 💪
For posture-based AI feedback apps using Python, FastAPI, and ML.

---

