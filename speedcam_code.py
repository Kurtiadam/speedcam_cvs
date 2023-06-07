from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
from sort import *
import os
import sys
import tesserocr
from PIL import Image
import matplotlib.pyplot as plt
api = tesserocr.PyTessBaseAPI()


class TrafficSpeedCamera:
    def __init__(self, input_path, input_mode):
        self.io_handler = IOHandler(input_path)
        self.vehicle_detector = VehicleDetector()
        self.oc_recognizer = OCR()
        self.image_enhancer = LPEnhancer()
        self.license_plate_detector = LicensePlateDetector(self.oc_recognizer, self.image_enhancer)
        self.object_tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator()
        self.fps_count = 0
        self.input_mode = input_mode
        self.input_path = input_path
        self.last_frame_time = time.time()
        self.distance_setup_ran = False

    def process_frame(self, frame, show_tracking):
        show_frame = frame.copy()
        vehicle_detections = self.vehicle_detector.detect_vehicles(show_frame)
        tracked_vehicles = self.object_tracker.track_objects(show_frame, vehicle_detections, show_tracking)
        if len(vehicle_detections) != 0:
            license_plates = self.license_plate_detector.detect_license_plates(frame, show_frame, tracked_vehicles)
        speeds = self.speed_estimator.estimate_speed(frame, tracked_vehicles)
        self.measure_fps(show_frame)
        cv2.imshow("Frame", show_frame)

    def run(self, show_tracking, distance_setup, ret = True):
        print("START HERE")
        if self.input_mode == "burst_photos":
            file_names = sorted(os.listdir(self.input_path))
            for file_name in file_names:
                image_path = os.path.join(self.input_path, file_name)
                frame = cv2.imread(image_path)
                if not self.distance_setup_ran:
                    clicks = []
                    fig, ax = plt.subplots(figsize=(16, 9))
                    ax.imshow(frame)
                    cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.speed_estimator.setup_distance(event, frame, clicks, cid, fig))
                    plt.show()
                    self.distance_setup_ran = True
                self.process_frame(frame, show_tracking)
                self.io_handler.check_end_stream(ret)
            ret = False
            self.io_handler.check_end_stream(ret)
            print("END HERE")

        elif self.input_mode == "video":
            while True:
                ret, frame = self.io_handler.cap.read()
                if not self.distance_setup_ran:
                    clicks = []
                    fig, ax = plt.subplots(figsize=(16, 9))
                    ax.imshow(frame)
                    cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.speed_estimator.setup_distance(event, frame, clicks, cid, fig))
                    plt.show()
                    self.distance_setup_ran = True
                self.process_frame(frame)
                cv2.imshow("Frame", frame)
                self.io_handler.check_end_stream(ret)

        else:
            raise NotImplementedError

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
        print("VEHICLE DETECTION HERE")
        detections = np.empty((0,5))
        # img_region = cv2.bitwise_and(frame, self.mask)
        vd_results = self.model_vd(frame, stream=True, classes=self.searched_class_indices, conf=0.7, iou=0.9, agnostic_nms = True)
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
        self.tracker = Sort(max_age=60, min_hits=5, iou_threshold=0.4)
        self.vehicle_preds = {}

    def track_objects(self, frame, detections, show):
        print("TRACKING HERE")
        track_results = self.tracker.update(detections)
        # [x1,y1,x2,y2,idx] - X coordinate from left, Y coordinate from top

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
                    self.vehicle_preds[idx]['speed'] = 0
        
            
            # if conf > self.vehicle_preds[idx]['vd_conf']:
                # self.vehicle_preds[idx]['vd_conf'] = conf
            self.vehicle_preds[idx]['vd_bbox_coords'] = bboxes_int
            self.vehicle_preds[idx]['tracked'] = True
            # Draw travelled track lines
            if show:
                self.vehicle_preds[idx]['vd_center'].append(center)
                if idx % 2 == 0:
                    R = 255
                    B = 0
                    G = 0
                else:
                    R = 0
                    B = 0
                    G = 255
                for i in range(len(self.vehicle_preds[idx]['vd_center']) - 1):
                    center_x = self.vehicle_preds[idx]['vd_center'][i]
                    center_x = tuple(map(int, center_x))
                    center_y = self.vehicle_preds[idx]['vd_center'][i + 1]
                    center_y = tuple(map(int, center_y))
                    cv2.line(frame, center_y, center_x, (B, G, R), thickness=2)
                cv2.putText(frame, f'{str(idx)}', [bboxes_int[2], bboxes_int[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (B, G, R), 2)

            else:
                self.vehicle_preds[idx]['vd_center'] = center
                cv2.putText(frame, f'{str(idx)}', [bboxes_int[2], bboxes_int[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return self.vehicle_preds



class LicensePlateDetector:
    def __init__(self, oc_recognizer, image_enhancer):
        self.model_lp = YOLO('./runs/detect/train5/weights/best.pt')
        self.oc_recognizer = oc_recognizer
        self.image_enhancer = image_enhancer

    def detect_license_plates(self, frame, show_frame, vehicle_preds):
        for idx in vehicle_preds.keys():
            bbox_vd = vehicle_preds[idx]['vd_bbox_coords']
            # [x1,y1,x2,y2,idx]
            print("LICENSE PLATE DETECTION HERE")
            cropped_vehicle = np.array(frame[bbox_vd[1]:bbox_vd[3], bbox_vd[0]:bbox_vd[2]])
            # Visualizing vehicles
            if vehicle_preds[idx]['tracked'] == True:
                cv2.imshow(str(idx), cropped_vehicle)
                x_w, y_w, _, h_w = cv2.getWindowImageRect(str(idx))
                lp_text = vehicle_preds[idx]['lp_text']
                cv2.putText(show_frame, lp_text, [bbox_vd[0],bbox_vd[3]+25], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                vehicle_preds[idx]['tracking_window_opened'] = True
            elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                cv2.destroyWindow(str(idx))
                if cv2.getWindowProperty(str(idx) + " license plate", cv2.WND_PROP_VISIBLE) >=1:
                    cv2.destroyWindow(str(idx) + " license plate")
                vehicle_preds[idx]['tracking_window_opened'] = False
            if vehicle_preds[idx]['tracked']:
                lp_preds = self.model_lp(cropped_vehicle, imgsz=320, iou=0.5)
                for result in lp_preds:
                            boxes = result.boxes
                            for box in boxes:
                                if len(box.xyxy[0]) != 0:
                                    x1_lp, y1_lp, x2_lp, y2_lp = tuple(map(int, box.xyxy[0]))
                                    bbox_arr_lp = np.array([x1_lp, y1_lp, x2_lp, y2_lp])
                                    lp_center = (int(bbox_vd[0] + (x1_lp+x2_lp)/2),int(bbox_vd[1] + (y1_lp+y2_lp)/2))
                                    vehicle_preds[idx]['lp_bbox_coords'] = bbox_arr_lp
                                    cropped_lp = np.array(cropped_vehicle[y1_lp:y2_lp, x1_lp:x2_lp])
                                    conf = math.ceil((box.conf[0]*100))/100
                                    vehicle_preds[idx]['lp_conf'] = conf

                                    # Visualizing license plates
                                    if vehicle_preds[idx]['tracked'] == True:
                                        cv2.imshow(str(idx) + " license plate", cropped_lp)
                                        cv2.moveWindow(str(idx) + " license plate", x_w, y_w + h_w)

                                        vehicle_preds[idx]['tracking_window_opened'] = True
                                    elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                                        cv2.destroyWindow(str(idx) + " license plate")
                                        vehicle_preds[idx]['tracking_window_opened'] = False
                                    
                                    if lp_center[0] >= (frame.shape[1]*0.05):
                                        enhanced_frame = self.image_enhancer.enhance_image(cropped_lp)
                                        lp_text, ocr_conf = self.oc_recognizer.read_license_plate(enhanced_frame)
                                        if vehicle_preds[idx]['lp_text'] == "" and 7 <= len(lp_text) <= 8:
                                            vehicle_preds[idx]['lp_text'] = lp_text
                                            vehicle_preds[idx]['ocr_conf'] = ocr_conf
                                        elif ocr_conf > vehicle_preds[idx]['ocr_conf'] and 7 <= len(lp_text) <= 8:
                                            vehicle_preds[idx]['lp_text'] = lp_text
                                            vehicle_preds[idx]['ocr_conf'] = ocr_conf
                                    else:
                                        print(idx, "OCR skipped - LP out of frame")

        for key, value in vehicle_preds.items():
            print(key, value)
        print("\n")

        return vehicle_preds





class LPEnhancer:
    def __init__(self) -> None:
        self.rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

    def enhance_image(self, frame):
        h,w = frame.shape[:2]
        test_lp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bil = cv2.bilateralFilter(test_lp_gray, 11, 17, 17)
        # cv2.imshow("2 - Bilateral Filter", bil)
        blackhat = cv2.morphologyEx(test_lp_gray, cv2.MORPH_BLACKHAT, self.rectKern)
        # cv2.imshow("Blackhat", blackhat)
        # equalized_image = cv2.equalizeHist(blackhat)
        # cv2.imshow("eq", equalized_image)
        # Angle correction
        edges = cv2.Canny(bil, 10, 50, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

        # if lines is not None:
        #     for line in lines:
        #         rho, theta = line[0]
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * (a))
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * (a))
                
        #         cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow("With lines", frame)

        angles = []
        try:
            for line in lines:
                rho, theta = line[0]
                angle = np.rad2deg(theta)
                angles.append(angle)
            predominant_angle = np.median(angles)
        except:
            predominant_angle = 90
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2),predominant_angle-90, 1)
        angle_corrected = cv2.warpAffine(blackhat, rotation_matrix, (w, h))

        # test_lp_gray_bw = cv2.threshold(angle_corrected, 0, 255,cv2. THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # enhanced_frame = cv2.erode(test_lp_gray_bw, kernel, iterations=1)
        enhanced_frame = angle_corrected
        # cv2.imshow("Enhanched", enhanced_frame)

        return enhanced_frame


class OCR():
    def __init__(self) -> None:
        self.alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        # self.options = "-c tessedit_char_whitelist={}".format(self.alphanumeric)
        # self.options += " --psm {}".format(7)
    
    def read_license_plate(self, frame):
        # data = pt.image_to_data(frame, config=self.options, output_type=Output.DICT)
        # confidences = data['conf']
        # confidences = [float(c) for c in confidences if c != '-1']
        # conf = np.mean(confidences) / 100
        # text = data['text']
        # text = ' '.join(text).strip()

        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_LINE) as api:
            api.SetVariable('tessedit_char_whitelist', self.alphanumeric)
            image = Image.fromarray(frame)
            api.SetImage(image)
            text = api.GetUTF8Text()
            text = text.strip()
            confidences = api.AllWordConfidences()
            confidences = [float(c) for c in confidences]
            conf = np.mean(confidences) / 100
            print("TESSEROCR PRED: ", text, "CONF: ", conf)
        return text, conf


class SpeedEstimator:
    def __init__(self):
        # (1146x894) bottom left lane piece - (1470,358) upper right lane piece
        self.dist_in_meters = 0.0
        self.dist_in_pixels = 0.0
        self.pixel_meter_ratio = 0.0
        self.time_per_frame = 0.18
        self.moving_average_size = 3

    def estimate_speed(self, frame, tracked_objects):
        for idx in tracked_objects.keys():
            total = 0
            num_frames = min(self.moving_average_size, len(tracked_objects[idx]['vd_center']))
            if num_frames >= 2:
                for i in range(num_frames-1):
                    travelled_distance_pixels = np.linalg.norm(np.array(tracked_objects[idx]['vd_center'][i+1]) - np.array(tracked_objects[idx]['vd_center'][i]))
                    total += travelled_distance_pixels
                travelled_distance_meters = total / self.pixel_meter_ratio
                speed = travelled_distance_meters / (self.time_per_frame*(num_frames-1))
                tracked_objects[idx]['speed'] = speed*3.6
            print('IDX',idx, 'EST SPEED', tracked_objects[idx]['speed'], "km/h")
        return tracked_objects
    
    def setup_distance(self, event, frame, clicks, cid, fig):
        if event.xdata is not None and event.ydata is not None:
            clicks.append((event.xdata, event.ydata))
            if len(clicks) == 2:
                p1, p2 = clicks
                self.dist_in_pixels = np.linalg.norm(np.array(p1) - np.array(p2))
                fig.canvas.mpl_disconnect(cid)
                plt.imshow(frame)
                plt.scatter(*zip(*clicks), color='red', marker='x')
                plt.show()
                dist_in_meters = input("\nHow many meters is this in reality?")
                self.dist_in_meters = float(dist_in_meters)
                self.pixel_meter_ratio = self.dist_in_pixels/self.dist_in_meters
                print(f"Pixel per meter conversion ratio is: {self.pixel_meter_ratio}" + " pixels/meter")


class IOHandler:
    def __init__(self, input_path):
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path)


    def check_end_stream(self, ret):
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit()


def main():
    speed_camera = TrafficSpeedCamera(r"C:\Users\Adam\Desktop\speedcam_samples\06.03\FAST_2", "burst_photos")
    speed_camera.run(show_tracking = True, distance_setup = True)


if __name__ == '__main__':
    main()