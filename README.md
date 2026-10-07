# Real-Time Robbery & Threat Detection System

A production-grade, cascaded dual-stage AI architecture designed to detect **Robbery, Intruders, Weapons (Guns/Knives), Face Masks (Ski Masks/Balaclavas), Surrender Poses (Hands-Up), and Restricted Counter Zone Violations** at **30 FPS real-time speed**.

---

## 📁 Project Architecture

```
E:\Aitronix Projects\Robber Detection\
│
├── stage_1_scanner/              # Stage 1: Fast Frame, Pose & Motion Scanner
│   ├── pose_engine.py           # YOLOv8-pose 17-skeletal keypoint tracking (~100+ FPS)
│   ├── florence_scanner.py      # Microsoft Florence-2-base open-vocab scanner (~120 ms)
│   └── zone_manager.py          # Polygon spatial area manager (Point-in-Polygon)
│
├── stage_2_verifier/             # Stage 2: High-Precision Verification
│   └── locate_anything_verifier.py # NVIDIA Locate Anything 3B GGUF HTTP client
│
├── stage_3_alert/                # Stage 3: Threat Evaluation & Alert Handling
│   ├── threat_evaluator.py      # Multi-factor threat scoring & 3-frame persistence buffer
│   └── alert_manager.py         # Json incident logging & evidence frame saver
│
├── models/                       # Model weights cache (Stored on local E: drive)
│   └── models--microsoft--Florence-2-base/
│
├── incident_logs/                # Saved threat evidence frames & JSON logs
├── zones.json                    # Polygon zone definitions
├── main_robber_detector.py       # Main real-time application entry point
└── README.md                     # Project documentation
```

---

## 🚀 How to Run the System

### 1. Launch the NVIDIA Locate Anything 3B Backend Server
Ensure your local server is running on port `8085`:
```powershell
& "M:\Programs\llama.cpp\prism llama\llama.cpp\build\bin\Release\llama-server.exe" -m "M:\AI\Models\Nvidia-locate Everything\LocateAnything-3B-Q4_K_M.gguf" -mm "M:\AI\Models\Nvidia-locate Everything\mmproj-LocateAnything-3B-BF16.gguf" -ngl 99 -fa on -c 16384 --chat-template chatml --port 8085 --host 127.0.0.1 --mlock --cache-reuse 256 --fit off --no-mmap --special
```

### 2. Run Main Detector Application

#### **Live Webcam (Default Camera 0):**
```bash
python main_robber_detector.py
```

#### **Run on a Video File:**
Inside [main_robber_detector.py](file:///e:/Aitronix%20Projects/Robber%20Detection/main_robber_detector.py), call:
```python
app.run("path/to/security_video.mp4", save_output=True)
```

---

## 🎮 Interactive Features During Video Stream
* **Press `z`**: Adds a default `cash_counter` polygon restricted zone.
* **Press `q`**: Safely quits the application and closes video windows.
