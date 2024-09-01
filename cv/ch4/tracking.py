from ultralytics import YOLO
import cv2

model = YOLO('yolov10m.pt')

video_path = './sample.mp4'
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist = True)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLO Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

