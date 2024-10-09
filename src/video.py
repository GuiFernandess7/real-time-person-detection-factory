from ultralytics import YOLO
import numpy as np
import cv2

url = ...

model = YOLO('yolo/yolov8s.pt')

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Erro ao abrir o stream de vídeo.")
    exit()

window_name = 'Stream MJPEG'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()

    if not success:
        print("Não foi possível ler o frame.")
        break

    results = model.track(frame, persist=True)

    filtered_results = []
    for result in results:
        if len(results) > 0:
            if len(result.boxes.cls) > 0 and result.boxes.cls[0] == 0:
                filtered_results.append(result)

    if len(results) > 0:
        frame_ = results[0].plot()
    else:
        frame_ = frame

    cv2.resizeWindow(window_name, 800, 600)
    cv2.imshow(window_name, frame_)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
