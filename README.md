# 🚨 Smart Campus Fire & Smoke Detection System

A real-time computer vision surveillance pipeline for detecting fire and smoke hazards on campus. The system runs a custom-trained **YOLOv8** object detection model behind a **FastAPI** backend, logs inference metadata to **MongoDB**, stores annotated evidence frames in **Supabase Cloud Storage**, sends **SMTP email alerts** with attached evidence, and exposes a **Streamlit** dashboard for live monitoring and analytics.

---

## 🏗️ Architecture & Data Flow

```
   [ CCTV Camera / Client Upload ]
                  │
                  ▼
         [ Streamlit Frontend ]
                  │
          (POST Image Stream)
                  │
                  ▼
          [ FastAPI Backend ]
                  │
         ┌────────┴────────┬─────────────────────┐
         ▼                 ▼                      ▼
  [ YOLOv8 Engine ]  [ MongoDB Cluster ]  [ Supabase Cloud Bucket ]
 (Bounding Boxes)    (Inference Logs)     (Evidence Frame Storage)
         │                 │                      │
         │                 ▼                      ▼
         │          [ Analytics UI ]     [ Public Asset URL ]
         │
         ▼
[ Background Worker Thread ]
         │
         ▼
[ Secure SMTP TLS Gateway ] ──► [ Security Alert Email ]
```

---

## ⚡ Key Features

- **Custom Object Detection** — YOLOv8 model fine-tuned to detect `fire` and `smoke` as separate classes, with bounding boxes and confidence scores.
- **Async API Layer** — FastAPI backend with non-blocking `BackgroundTasks` so email alerts and logging never block the inference response.
- **Cloud Evidence Storage** — Annotated frames are uploaded to Supabase Storage, generating a permanent, shareable URL per detection.
- **Persistent Inference Logs** — Every prediction (detections, confidence, latency, timestamp) is written to MongoDB via the async `motor` driver.
- **Live Dashboard** — A Streamlit UI with webcam/upload input, a threat detection timeline, latency charts, and raw log inspection.
- **Automated Email Alerts** — SMTP notifications with the annotated image attached are dispatched whenever a frame is processed.

---

## 📂 Repository Structure

```
├── yolov8n.pt              # Base YOLOv8 detection weights (pretrained)
├── yolov8n-cls.pt          # Base YOLOv8 classification weights (pretrained)
├── download_dataset.py     # Pulls the labeled dataset from Roboflow
├── organize_dataset.py     # Sorts loose images into train/val fire & smoke folders
├── train_model.py          # Trains the custom YOLOv8 detection model
├── train_classifier.py     # Trains a YOLOv8 classification model variant
├── export_model.py         # Exports trained weights (best.pt) to ONNX
├── detect.py                # CLI script for running batch inference on a folder/image
├── interface.py             # Standalone CLI inference runner (alternate entry point)
├── main.py                  # FastAPI backend: inference endpoint, alerts, logging
├── frontend.py               # Streamlit dashboard: live upload + analytics
├── database.py               # Async MongoDB connection & log persistence
├── storage.py                 # Supabase cloud storage upload logic
├── requirements.txt            # Python dependencies
└── README.md                    # This file
```

Trained weights are expected at `runmodel/detect/train/weights/best.pt` after running `train_model.py`; the backend and CLI scripts will fall back to `yolov8n.pt` if no custom weights are found.

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/SmartCampusFireSmokeDetection.git
cd SmartCampusFireSmokeDetection
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

This project requires credentials for MongoDB, Supabase, Roboflow, and SMTP email. **Do not hardcode these in source files.** Create a `.env` file in the project root:

```env
MONGO_DETAILS=mongodb://localhost:27017
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-anon-key
ROBOFLOW_API_KEY=your-roboflow-api-key
SENDER_EMAIL=your-alert-sender@gmail.com
SENDER_PASSWORD=your-gmail-app-password
RECEIVER_EMAIL=your-alert-recipient@example.com
```

> ⚠️ **Security note:** Add `.env` to `.gitignore` before your first commit. Never commit real API keys, database URLs, or email app passwords to a public repository — rotate any credentials that have already been exposed.

### 5. Start MongoDB

Make sure a local MongoDB instance is running at `mongodb://localhost:27017` (or update `MONGO_DETAILS` to point at your own cluster).

---

## 🧠 Training Your Own Model (optional)

If you want to retrain the detector on your own data instead of using the provided weights:

```bash
python download_dataset.py     # pulls the labeled dataset from Roboflow
python organize_dataset.py     # only needed if working from a flat image folder
python train_model.py          # trains the YOLOv8 detector
python export_model.py         # optional: export best.pt to ONNX
```

Trained weights are saved to `runmodel/detect/train/weights/best.pt`.

---

## ▶️ Running the System

Run the backend and frontend in separate terminals (with the virtual environment activated in both).

**Terminal 1 — API backend:**
```bash
python main.py
```
Starts the FastAPI server at `http://127.0.0.1:8000`, loads the YOLO model, and connects to MongoDB.

**Terminal 2 — Dashboard:**
```bash
streamlit run frontend.py
```
Opens the dashboard in your browser, usually at `http://localhost:8501`.

---

## 🧪 Using the Dashboard

1. Go to the **Live Camera Stream & Upload** tab.
2. Upload an image or capture a frame from your webcam.
3. The frame is sent to the backend for inference; detected objects, confidence scores, and the annotated image are displayed.
4. Switch to the **System Analytics & Logs** tab and click **Sync & Refresh Analytics Dashboard** to view historical inference logs, detection timelines, and latency metrics pulled from MongoDB.

---

## 🔌 API Reference

### `POST /predict-frame`
Accepts a multipart image file, runs detection, uploads the annotated frame to Supabase, logs the result to MongoDB, and triggers an email alert.

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "frame.jpg",
    "cloud_url": "https://.../evidence.jpg",
    "timestamp": 1735689600.0,
    "total_objects_detected": 1,
    "inference_time_seconds": 0.842,
    "detections": [
      {"object_type": "fire", "confidence": 0.87, "bbox": [x1, y1, x2, y2]}
    ]
  }
}
```

### `GET /fetch-logs`
Returns the 10 most recent inference logs from MongoDB.

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for full pinned versions):

- `fastapi`, `uvicorn` — API server
- `ultralytics` — YOLOv8 model
- `motor`, `pymongo` — MongoDB async driver
- `supabase` — cloud storage client
- `streamlit` — dashboard UI
- `roboflow` — dataset download for training
- `Pillow`, `torch`, `torchvision` — image processing & model backend

---

