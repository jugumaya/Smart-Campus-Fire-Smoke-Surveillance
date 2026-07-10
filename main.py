import io
import time
import smtplib
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import uvicorn

# Ingest internal custom integration interfaces
from database import save_inference_log, logs_collection
from storage import upload_image_to_cloud

app = FastAPI(
    title="Smoke & Fire Detection Surveillance Pipeline", 
    version="3.6.0"
)

# Enforce open validation protocols for dashboard connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secured automated notification client access configuration tokens
SENDER_EMAIL = "jugumayasaha@gmail.com"       
SENDER_PASSWORD = "wtyg eulx xnva xxru" 
RECEIVER_EMAIL = "jugumayasahad@gmail.com"   

def send_security_alert(object_type: str, confidence: float, image_bytes: bytes):
    """
    Dispatches secure SMTP text alert sequences along with visual attachments.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = f"🚨 CRITICAL ALARM: {object_type.upper()} Threat Ingestion Registered on Campus!"

        body = (
            f"⚠️ AUTOMATED SECURITY ALERT: Campus Network Security Breach!\n\n"
            f"Threat Vector Isolated: {object_type.upper()}\n"
            f"Confidence Score Metric: {confidence * 100:.2f}%\n"
            f"System Ingestion Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n"
            f"Action Protocol: Immediate onsite isolation required. Verify live evidence data attachment immediately."
        )
        msg.attach(MIMEText(body, 'plain'))

        # Embed structural frame proof into the outbound payload data stream
        img_attachment = MIMEImage(image_bytes, name="evidence.jpg")
        msg.attach(img_attachment)

        # Authenticate and route transmission block via secure TLS channels
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"✅ [ALERT DISPATCH] Message notification securely transmitted for class: {object_type}")
    except Exception as e:
        print(f"❌ [ALERT SYSTEM FAULT] Outbound message termination encountered: {str(e)}")

def _load_model() -> YOLO:
    """
    Evaluates directory structures sequentially to lock down production parameters.
    Falls back gracefully onto foundational models to avoid initialization faults.
    """
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "runs" / "detect" / "runmodel" / "detect" / "train" / "weights" / "best.pt",
        base_dir / "runmodel" / "detect" / "train" / "weights" / "best.pt",
        base_dir / "best.pt",
        base_dir / "yolov8n.pt" 
    ]

    for p in candidates:
        if p.exists() or p.name == "yolov8n.pt":
            try:
                m = YOLO(str(p))
                print(f"✅ Production Weights Successfully Engaged From: {p}")
                return m
            except Exception as e:
                print(f"⚠️ Initialization anomaly encountered matching configuration variables at {p}: {e}")

    return YOLO("yolov8n.pt")

model = _load_model()

@app.post("/predict-frame")
async def predict_frame(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Primary processing endpoint executing frame classification, asynchronous asset distribution,
    historical logging entries, and notification task routing.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid processing stream. Target resource must be an image.")
    
    start_time = time.time()
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Enforced low-threshold validation logic to handle tracking during active debugging phases
        results = model(image, conf=0.25)
        
        detections = []
        trigger_alert = False
        alert_object = "Surveillance Object"
        alert_conf = 0.01

        # Unpack prediction coordinates and detection metrics
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                obj_name = model.names[class_id]
                conf = round(float(box.conf[0]), 4)
                
                detections.append({
                    "object_type": obj_name,
                    "confidence": conf,
                    "bbox": [round(float(x), 2) for x in box.xyxy[0]]
                })
                
                trigger_alert = True
                if conf > alert_conf:
                    alert_object = obj_name
                    alert_conf = conf

        # Superimpose tracking bounding annotations back onto standard image arrays
        plotted_bgr = results[0].plot()
        plotted_image = Image.fromarray(plotted_bgr[..., ::-1]) 
        
        buffer = io.BytesIO()
        plotted_image.save(buffer, format="JPEG", quality=85)
        processed_image_bytes = buffer.getvalue()
        
        # Route processing frames to storage pipelines asynchronously
        filename = file.filename or "processed_image.jpg"
        cloud_image_url = await upload_image_to_cloud(processed_image_bytes, filename)
        
        # Deploy alert handling processing queues in background worker threads
        if len(detections) > 0:
            background_tasks.add_task(send_security_alert, alert_object, alert_conf, processed_image_bytes)
        else:
            background_tasks.add_task(send_security_alert, "System General Scan", 1.0, processed_image_bytes)
        
        latency = round(time.time() - start_time, 4)
        
        # Generate complete tracing schema for database ingestion
        pipeline_output = {
            "filename": file.filename,
            "cloud_url": cloud_image_url if cloud_image_url else "Cloud Storage Connection Loss",
            "timestamp": time.time(),
            "total_objects_detected": len(detections),
            "inference_time_seconds": latency,
            "detections": detections,
            "alert_triggered": True 
        }
        
        # Record structured telemetry data safely in database collections
        await save_inference_log(pipeline_output)
        
        # Stringify object identifiers to keep transmission payload JSON serializable
        if "_id" in pipeline_output:
            pipeline_output["_id"] = str(pipeline_output["_id"])
        
        return JSONResponse(status_code=200, content={"success": True, "data": pipeline_output})
        
    except Exception as e:
        print(f"❌ Structural Failure Inside Ingestion Endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "detail": str(e)})

@app.get("/fetch-logs")
async def get_all_logs():
    """
    Queries historical database log profiles to supply data points to analytics modules.
    """
    try:
        logs = []
        async for log in logs_collection.find().sort("timestamp", -1).limit(10):
            if "_id" in log:
                log["_id"] = str(log["_id"])
            logs.append(log)
        return {"status": "success", "total_records": len(logs), "data": logs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    