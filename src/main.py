from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import Request
from ultralytics import YOLO
import cv2
import os
import dotenv
import asyncio

from classes import class_names

dotenv.load_dotenv()

running = True

app = FastAPI()
url = os.getenv('URL_ADDRESS')
model = YOLO('yolo/yolov8s.pt')
window_name = 'Stream MJPEG'

@app.get("/detections")
async def detections(request: Request):
    global running
    cap = cv2.VideoCapture(url)

    async def event_stream():
        global running
        try:
            while running:
                success, frame = cap.read()
                if not success:
                    break

                results = model.track(frame, persist=True, verbose=False)
                detected_objects = {}

                if len(results) > 0:
                    for result in results:
                        if len(result.boxes.cls) > 0:
                            for cls in result.boxes.cls:
                                obj_name = class_names.get(int(cls))
                                if obj_name == 'person':
                                    if obj_name in detected_objects:
                                        detected_objects[class_names.get(int(cls))] += 1
                                    else:
                                        detected_objects[class_names.get(int(cls))] = 1

                yield f"data: {detected_objects}\n\n"
                await asyncio.sleep(0.1)

        finally:
            cap.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.on_event("shutdown")
async def shutdown_event():
    global running
    running = False

@app.get("/video")
async def video():
    cap = cv2.VideoCapture(url)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, persist=True, verbose=False)

            if results:
                for result in results:
                    if len(result.boxes.cls) > 0:
                        if any(int(cls) == 0 for cls in result.boxes.cls):
                            frame = result.plot()

            _, buffer = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')
