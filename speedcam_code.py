from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import csv
from sort import *
from matplotlib import pyplot as plt
from varname import nameof


class TrafficSpeedCamera:
    def __init__(self, video_path):
        self.video_io_handler = VideoIOHandler(video_path)
        self.vehicle_detector = VehicleDetector()
        self.license_plate_detector = LicensePlateDetector()
        self.object_tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator()
        self.fps_count = 0
        self.last_frame_time = time.time()

    def run(self):
        while True:
            ret, frame = self.video_io_handler.cap.read()

            # inph, inpw = frame.shape[:2]
            # frame_ds = cv2.resize(frame, [int(inpw/4), int(inph/4)])

            vehicle_detections = self.vehicle_detector.detect_vehicles(frame)
            tracked_vehicles = self.object_tracker.track_objects(frame, vehicle_detections)
            license_plates = self.license_plate_detector.detect_license_plates(frame, tracked_vehicles)
            speeds = self.speed_estimator.estimate_speed(frame, tracked_vehicles)
            self.measure_fps(frame)
            
            cv2.imshow("Frame", frame)
            self.video_io_handler.check_end_stream(ret)

    def measure_fps(self, frame):
        curr_time = time.time()
        frame_time = curr_time-self.last_frame_time
        fps = int(1/frame_time)
        cv2.putText(frame, str(fps), [frame.shape[1]-100, frame.shape[0]-50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.last_frame_time = time.time()


class VehicleDetector:
    def __init__(self):
        self.model_vd = YOLO('./yolov8s.pt')
        # ["bicycle", "car", "motorcycle", "bus", "truck"]
        self.searched_class_indices = [1, 2, 3, 5, 7]
        self.mask = cv2.imread('mask.png')

    def detect_vehicles(self, frame):
        detections = np.empty((0,5))
        img_region = cv2.bitwise_and(frame, self.mask)
        vd_results = self.model_vd(frame, stream=True, classes=self.searched_class_indices, conf=0.3)
        print("VD reSULTS: ", vd_results)
        for result in vd_results:
            boxes = result.boxes
            for box in boxes:
                x1_vd, y1_vd, x2_vd, y2_vd = tuple(map(int, box.xyxy[0]))
                cv2.rectangle(frame, (x1_vd, y1_vd),(x2_vd, y2_vd), (255, 0, 255), 2)
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                cv2.putText(frame, f'{str(self.model_vd.model.names[cls])} {conf}', [x1_vd, y1_vd], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                bbox_arr = np.array([x1_vd, y1_vd, x2_vd, y2_vd, conf])
                detections = np.vstack((detections, bbox_arr))

        return detections


class LicensePlateDetector:
    def __init__(self):
        self.model_lp = YOLO('./runs/detect/train5/weights/best.pt')

    def detect_license_plates(self, frame, vehicle_preds):
        print("PRED:", vehicle_preds)
        for idx in vehicle_preds.keys():
            print("IDX:", idx)
            bboxes = vehicle_preds[idx]['vd_bbox_coords']
            cropped_vehicle = frame[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]]

            # Visualizing vehicles
            if vehicle_preds[idx]['tracked'] == True:
                cv2.imshow(str(idx), cropped_vehicle)
                vehicle_preds[idx]['tracking_window_opened'] = True
            elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                cv2.destroyWindow(str(idx))
                vehicle_preds[idx]['tracking_window_opened'] = False

            lp_preds = self.model_lp(cropped_vehicle, imgsz=320)
            for result in lp_preds:
                        boxes = result.boxes
                        for box in boxes:
                            if len(box.xyxy[0]) != 0:
                                print("BOX LEN: ", len(box.xyxy[0]))
                                x1_lp, y1_lp, x2_lp, y2_lp = tuple(map(int, box.xyxy[0]))
                                bbox_arr_lp = np.array([x1_lp, y1_lp, x2_lp, y2_lp])
                                print("BBOX ArrAY: ", bbox_arr_lp)
                                vehicle_preds[idx]['lp_pred'] = bbox_arr_lp
                                cropped_lp = cropped_vehicle[y1_lp:y2_lp, x1_lp:x2_lp]

                                conf = math.ceil((box.conf[0]*100))/100
                                print("CONF: ", conf)

                                # Visualizing license plates
                                if vehicle_preds[idx]['tracked'] == True:
                                    cv2.imshow(str(idx) + " license plate", cropped_lp)
                                    vehicle_preds[idx]['tracking_window_opened'] = True
                                elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                                    cv2.destroyWindow(str(idx) + " license plate")
                                    vehicle_preds[idx]['tracking_window_opened'] = False
        return vehicle_preds


class ObjectTracker:
    def __init__(self):
        self.tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)
        self.vehicle_preds = {}

    def track_objects(self, frame, detections):
        # Draw travelled track lines
        # for i in range(len(vehicle_preds[idx]['centers']) - 1):
        #     center_x = vehicle_preds[idx]['centers'][i]
        #     center_x = tuple(map(int, center_x))
        #     center_y = vehicle_preds[idx]['centers'][i + 1]
        #     center_y = tuple(map(int, center_y))
        #     cv2.line(frame, center_y, center_x, (255, 0, 0), thickness=2)
        # print("DETS: ", detections)
        track_results = self.tracker.update(detections)
        # [x1,y1,x2,y2,idx]

        for vehicle in self.vehicle_preds:
             self.vehicle_preds[vehicle]['tracked'] = False

        for result in track_results:
            idx = int(result[-1])
            bboxes = result[:-1]
            bboxes_int = tuple(map(int, bboxes))
            center = ((int(result[2]+result[0])/2),int((result[3]+result[1])/2))
            
            if idx not in self.vehicle_preds:
                    self.vehicle_preds[idx] = {}
                    self.vehicle_preds[idx]['vd_center'] = []
                    self.vehicle_preds[idx]['vd_bbox_coords'] = []
                    self.vehicle_preds[idx]['vd_conf'] = 0
                    self.vehicle_preds[idx]['lp_bbox_coords'] = []
                    self.vehicle_preds[idx]['lp_conf'] = 0
                    self.vehicle_preds[idx]['tracked'] = False
                    self.vehicle_preds[idx]['tracking_window_opened'] = False
        
            
            # if conf > self.vehicle_preds[idx]['vd_conf']:
                # self.vehicle_preds[idx]['vd_conf'] = conf
            self.vehicle_preds[idx]['vd_bbox_coords'] = bboxes_int
            self.vehicle_preds[idx]['vd_center'] = center
            self.vehicle_preds[idx]['tracked'] = True

        return self.vehicle_preds


class SpeedEstimator:
    def __init__(self):
        # (1146x894) bottom left lane piece - (1470,358) upper right lane piece
        ref_points = np.array([[1470,358],[1146,894]])
        dist_in_meters = 1+1.75+1+1.75+1
        dist_in_pixels = math.sqrt(ref_points[0][0]-ref_points[1][0])**2+(ref_points[0][1]-ref_points[1][1])**2
        pixel_meter_ratio = dist_in_pixels/dist_in_meters

    def estimate_speed(self, frame, tracked_objects):
        pass

@staticmethod
class FPSMeasurment:
        def measure_fps(begin_time, frame):
            end_time = time.time()
            frame_time = end_time-begin_time
            fps = int(1/frame_time)
            cv2.putText(frame, str(fps), [frame.shape[0]-100, frame.shape[0]-50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


class VideoIOHandler:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def check_end_stream(self, ret):
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    speed_camera = TrafficSpeedCamera('./samples/Videok/P1010534.avi')
    speed_camera.run()


if __name__ == '__main__':
    main()