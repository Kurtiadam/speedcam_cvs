from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO('../yolov8x.pt')
cap = cv2.VideoCapture('samples\\road_footage.mp4')
searched_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
searched_classes_indices = [1,2,3,5,7]

while(cap.isOpened()):
    ret, frame = cap.read()
    inph, inpw = frame.shape[:2]
    frame_ds = cv2.resize(frame, [int(inpw/4),int(inph/4)])
    frame_ds = frame
    results = model(frame_ds, stream=True, classes = searched_classes_indices)
    bbox = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # bbox.append([x1,x2,y1,y2])
            cv2.rectangle(frame_ds, (x1,y1),(x2,y2),(255,0,255),2)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            cv2.putText(frame_ds, f'{str(model.model.names[cls])} {conf}', [x1,y1], cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    cv2.imshow('Frame', frame_ds)

    cropped = frame_ds[y1:y2,x1:x2]
    cv2.imshow('Cropped', cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()