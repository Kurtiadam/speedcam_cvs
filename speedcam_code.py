from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
from sort import *
import pytesseract as pt
from pytesseract import Output


class TrafficSpeedCamera:
    def __init__(self, video_path):
        self.video_io_handler = VideoIOHandler(video_path)
        self.vehicle_detector = VehicleDetector()
        self.oc_recognizer = OCR()
        self.image_enhancer = LPEnhancer()
        self.license_plate_detector = LicensePlateDetector(self.oc_recognizer, self.image_enhancer)
        self.object_tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator()
        self.fps_count = 0
        self.last_frame_time = time.time()

    def run(self):
        while True:
            ret, frame = self.video_io_handler.cap.read()
            inph, inpw = frame.shape[:2]
            frame_ds = cv2.resize(frame, [int(inpw/4), int(inph/4)])
            frame_ds = frame
            vehicle_detections = self.vehicle_detector.detect_vehicles(frame_ds)
            tracked_vehicles = self.object_tracker.track_objects(frame_ds, vehicle_detections)
            license_plates = self.license_plate_detector.detect_license_plates(frame_ds, tracked_vehicles)
            # speeds = self.speed_estimator.estimate_speed(frame, tracked_vehicles)
            self.measure_fps(frame_ds)
            
            cv2.imshow("Frame", frame_ds)
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
        self.mask = cv2.imread('mask_half.png')

    def detect_vehicles(self, frame):
        detections = np.empty((0,5))
        # img_region = cv2.bitwise_and(frame, self.mask)
        vd_results = self.model_vd(frame, stream=True, classes=self.searched_class_indices, conf=0.5, iou=0.9)
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


class ObjectTracker:
    def __init__(self):
        self.tracker = Sort(max_age=60, min_hits=10, iou_threshold=0.4)
        self.vehicle_preds = {}

    def track_objects(self, frame, detections):
        track_results = self.tracker.update(detections)
        # [x1,y1,x2,y2,idx]

        for vehicle in self.vehicle_preds:
            self.vehicle_preds[vehicle]['tracked'] = False

        for result in track_results:
            idx = int(result[-1])
            bboxes = result[:-1]
            bboxes_int = tuple(map(int, bboxes))
            bboxes_int = np.clip(bboxes_int, a_min=0, a_max=None) # Prevent any minus values
            center = (int((result[2]+result[0])/2),int((result[3]+result[1])/2))
            
            if idx not in self.vehicle_preds:
                    self.vehicle_preds[idx] = {}
                    self.vehicle_preds[idx]['vd_center'] = []
                    self.vehicle_preds[idx]['vd_bbox_coords'] = []
                    self.vehicle_preds[idx]['vd_conf'] = 0
                    self.vehicle_preds[idx]['lp_bbox_coords'] = []
                    self.vehicle_preds[idx]['lp_conf'] = 0
                    self.vehicle_preds[idx]['lp_text'] = ""
                    self.vehicle_preds[idx]['ocr_conf'] = 0
                    self.vehicle_preds[idx]['tracked'] = False
                    self.vehicle_preds[idx]['tracking_window_opened'] = False
        
            
            # if conf > self.vehicle_preds[idx]['vd_conf']:
                # self.vehicle_preds[idx]['vd_conf'] = conf
            self.vehicle_preds[idx]['vd_bbox_coords'] = bboxes_int
            self.vehicle_preds[idx]['vd_center'] = center
            self.vehicle_preds[idx]['tracked'] = True

            # # Draw travelled track lines
            # for i in range(len(self.vehicle_preds[idx]['vd_center']) - 1):
            #     center_x = self.vehicle_preds[idx]['vd_center'][i]
            #     center_x = tuple(map(int, center_x))
            #     center_y = self.vehicle_preds[idx]['vd_center'][i + 1]
            #     center_y = tuple(map(int, center_y))
            #     cv2.line(frame, center_y, center_x, (255, 0, 0), thickness=2)

        return self.vehicle_preds



class LicensePlateDetector:
    def __init__(self, oc_recognizer, image_enhancer):
        self.model_lp = YOLO('./runs/detect/train5/weights/best.pt')
        self.oc_recognizer = oc_recognizer
        self.image_enhancer = image_enhancer

    def detect_license_plates(self, frame, vehicle_preds):
        frame_h = int(frame.shape[0]/2)
        for idx in vehicle_preds.keys():
            bbox = vehicle_preds[idx]['vd_bbox_coords']
            # [x1,y1,x2,y2,idx]
            print("ASDASDA", vehicle_preds[idx]['vd_center'][1], frame_h)
            if vehicle_preds[idx]['vd_center'][1] < frame_h:
                cropped_vehicle = np.array(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])

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
                                    x1_lp, y1_lp, x2_lp, y2_lp = tuple(map(int, box.xyxy[0]))
                                    bbox_arr_lp = np.array([x1_lp, y1_lp, x2_lp, y2_lp])
                                    vehicle_preds[idx]['lp_bbox_coords'] = bbox_arr_lp
                                    cropped_lp = np.array(cropped_vehicle[y1_lp:y2_lp, x1_lp:x2_lp])

                                    conf = math.ceil((box.conf[0]*100))/100
                                    vehicle_preds[idx]['lp_conf'] = conf

                                    # Visualizing license plates
                                    if vehicle_preds[idx]['tracked'] == True:
                                        cv2.imshow(str(idx) + " license plate", cropped_lp)
                                        vehicle_preds[idx]['tracking_window_opened'] = True
                                    elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                                        cv2.destroyWindow(str(idx) + " license plate")
                                        vehicle_preds[idx]['tracking_window_opened'] = False
                                    
                                    enhanced_frame = self.image_enhancer.enhance_image(cropped_lp)
                                    cv2.imshow('Enchanced final', enhanced_frame)
                                    lp_text, ocr_conf = self.oc_recognizer.read_license_plate(enhanced_frame)

                                    if ocr_conf > vehicle_preds[idx]['ocr_conf']:
                                        vehicle_preds[idx]['lp_text'] = lp_text
                                        vehicle_preds[idx]['ocr_conf'] = ocr_conf
                print("\nPRED:", vehicle_preds)                        
            else:
                print("\nPRED:", vehicle_preds)
                continue

        return vehicle_preds





class LPEnhancer:
    def __init__(self) -> None:
        self.rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    def enhance_image(self, frame):
        w,h = frame.shape[:2]
        print("ASDASDA", w,h)
        test_lp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(test_lp_gray, cv2.MORPH_BLACKHAT, self.rectKern)
        # Angle correction
        edges = cv2.Canny(blackhat, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(blackhat, 1, np.pi / 180, 50)
        angles = []
        try:
            for line in lines:
                rho, theta = line[0]
                angle = np.rad2deg(theta)
                angles.append(angle)
            predominant_angle = np.median(angles)
        except:
            predominant_angle = 90
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2),predominant_angle-90, 1)
        angle_corrected = cv2.warpAffine(blackhat, rotation_matrix, (h, w))

        # test_lp_gray_bw = cv2.adaptiveThreshold(angle_corrected,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,1,5)
        enhanced_frame = frame
        # enhanced_frame = cv2.erode(test_lp_gray_bw, (2,2), iterations=1)

        return enhanced_frame


class OCR():
    def __init__(self) -> None:
        self.alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.options = "-c tessedit_char_whitelist={}".format(self.alphanumeric)
        self.options += " --psm {}".format(7)
    
    def read_license_plate(self, frame):
        data = pt.image_to_data(frame, config=self.options, output_type=Output.DICT)
        confidences = data['conf']
        confidences = [float(c) for c in confidences if c != '-1']
        conf = np.mean(confidences) / 100
        text = data['text']
        text = ' '.join(text).strip()
        return text, conf


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
        if cv2.waitKey(100) & 0xFF == ord('q') or not ret:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    speed_camera = TrafficSpeedCamera('./samples/Videok/second_try.avi')
    speed_camera.run()


if __name__ == '__main__':
    main()