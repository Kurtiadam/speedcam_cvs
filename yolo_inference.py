from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r'C:\Users\Adam\Desktop\best_10000_s.pt')
cap = cv2.VideoCapture('samples\highway_footage.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    inph, inpw = frame.shape[:2]
    frame_ds = cv2.resize(frame, [int(inpw/4),int(inph/4)])
    results = model(frame_ds, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_ds, (x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow('Frame', frame_ds)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()