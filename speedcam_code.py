from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
from sort import *
from matplotlib import pyplot as plt


def main():
    model_vd = YOLO('./yolov8s.pt')
    model_lp = YOLO('./runs/detect/train5/weights/best.pt')
    cap = cv2.VideoCapture('./samples/Videok/P1010534.avi')
    searched_class_indices = [1,2,3,5,7] # ["bicycle", "car", "motorcycle", "bus", "truck"]
    mask = cv2.imread('mask.png')
    vehicle_preds = {}

    # (1146x894) bottom left lane piece - (1470,358) upper right lane piece
    # ref_points = np.array([[1470,358],[1146,894]])
    # dist_in_meters = 1+1.75+1+1.75+1
    # dist_in_pixels = math.sqrt(ref_points[0][0]-ref_points[1][0])^2+(ref_points[0][1]-ref_points[1][1])^2
    # pixel_meter_ratio = dist_in_pixels/dist_in_meters

    # Tracking
    tracker = Sort(max_age = 60, min_hits=1, iou_threshold=0.3)

    while(cap.isOpened()):
        begin_time = time.time()
        ret, frame = cap.read()
        inph, inpw = frame.shape[:2]
        frame_ds = cv2.resize(frame, [int(inpw/4),int(inph/4)])
        frame_ds = frame
        img_region = cv2.bitwise_and(frame_ds, mask)
        vd_results = model_vd(frame_ds, stream=True, classes = searched_class_indices, conf=0.3)
        detections = np.empty((0,5))
        for result in vd_results:
            boxes = result.boxes
            for box in boxes:
                x1_vd, y1_vd, x2_vd, y2_vd = tuple(map(int, box.xyxy[0]))
                cv2.rectangle(frame_ds, (x1_vd,y1_vd),(x2_vd,y2_vd),(255,0,255),2)
                conf = math.ceil((box.conf[0]*100))/100
                print(conf)
                cls = int(box.cls[0])
                cv2.putText(frame_ds, f'{str(model_vd.model.names[cls])} {conf}', [x1_vd,y1_vd], cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                bbox_arr = np.array([x1_vd,y1_vd,x2_vd,y2_vd,conf])
                detections = np.vstack((detections, bbox_arr))

        # Saving bbox centers for each vehicle - [x1,y1,x2,y2,idx]
        track_results = tracker.update(detections)
        for result in track_results:
            idx = int(result[-1])
            bboxes = result[:-1]
            bboxes_int = tuple(map(int, bboxes))
            center = ((int(result[2]+result[0])/2),int((result[3]+result[1])/2))

            if idx not in vehicle_preds:
                vehicle_preds[idx] = {}
                vehicle_preds[idx]['centers'] = []
                vehicle_preds[idx]['bbox_coords'] = []
                vehicle_preds[idx]['cropped_vehicle'] = []
                vehicle_preds[idx]['lp_pred'] = []
                vehicle_preds[idx]['vd_conf'] = 0
            # vehicle_preds[idx]['centers'].append(center)
            if conf > vehicle_preds[idx]['vd_conf']:
                vehicle_preds[idx]['vd_conf'] = conf
                vehicle_preds[idx]['bbox_coords'] = bboxes_int
                cropped_vehicle = frame_ds[bboxes_int[1]:bboxes_int[3],bboxes_int[0]:bboxes_int[2]]
                lp_preds = model_lp(cropped_vehicle, imgsz=320)
                for result in lp_preds:
                    boxes = result.boxes
                    for box in boxes:
                        x1_lp, y1_lp, x2_lp, y2_lp = tuple(map(int, box.xyxy[0]))
                        bbox_arr_lp = np.array([x1_lp,y1_lp,x2_lp,y2_lp,conf])
                        vehicle_preds[idx]['lp_pred'] = bbox_arr_lp
                cropped_lp = cropped_vehicle[y1_lp:y2_lp,x1_lp:x2_lp]
            # vehicle_preds[idx]['cropped_vehicle'] = frame_ds[bboxes_int[1]:bboxes_int[3],bboxes_int[0]:bboxes_int[2]]

            # Draw travelled track lines
            # for i in range(len(vehicle_preds[idx]['centers']) - 1):
            #     center_x = vehicle_preds[idx]['centers'][i]
            #     center_x = tuple(map(int, center_x))
            #     center_y = vehicle_preds[idx]['centers'][i + 1]
            #     center_y = tuple(map(int, center_y))
            #     cv2.line(frame, center_y, center_x, (255, 0, 0), thickness=2)

        print(vehicle_preds)
        # FPS measurement
        end_time = time.time()
        frame_time = end_time-begin_time
        fps = int(1/frame_time)
        cv2.putText(frame_ds,str(fps),[inpw-100,inph-50],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # plt.imshow(frame_ds)
        # plt.show()
        cv2.imshow('Frame', frame_ds)
        cv2.imshow('LP', cropped_lp)
        # cv2.imshow('Zoomed',zoomed_vehicles)
        # cv2.imshow('Mask', img_region)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()