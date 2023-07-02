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
from typing import Tuple, Dict
api = tesserocr.PyTessBaseAPI()


class TrafficSpeedCamera:
    """Class for running the speed camera algorithm"""

    def __init__(self, input_path: str, input_mode: str, fps: int = 6):
        """
        Args:
            input_path (str): Path to the input file or directory.
            input_mode (str): Input mode, either "burst_photos" or "video".
            fps (int, optional): Frames per second of the source. Defaults to 6.
        """
        self.io_handler = IOHandler(input_path)
        self.vehicle_detector = VehicleDetector()
        self.oc_recognizer = OCR()
        self.image_enhancer = LPEnhancer()
        self.license_plate_detector = LicensePlateDetector(
            self.oc_recognizer, self.image_enhancer)
        self.object_tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator(fps)
        self.fps_count = 0
        self.input_mode = input_mode
        self.input_path = input_path
        self.last_frame_time = time.time()
        self.distance_setup_ran = False
        self.iter = 0

    def process_frame(self, frame: np.ndarray, show_tracking: bool, distance_setup: bool) -> None:
        """Process a single frame

        Args:
            frame (np.ndarray): The frame to process.
            show_tracking (bool): Flag indicating whether to display tracking information.
            distance_setup (bool): Flag indicating whether to setup distance measurement.
        """
        show_frame = frame.copy()
        vehicle_detections = self.vehicle_detector.detect_vehicles(show_frame)
        tracked_vehicles = self.object_tracker.track_objects(
            show_frame, vehicle_detections, show_tracking)
        if len(vehicle_detections) != 0:
            tracked_vehicles_lp = self.license_plate_detector.detect_license_plates(
                frame, show_frame, tracked_vehicles)
        if distance_setup:
            tracked_vehicles_lp_speeds = self.speed_estimator.estimate_speed(
                show_frame, tracked_vehicles, self.iter)
        self.measure_fps(show_frame)
        cv2.imshow("Frame", show_frame)

    def run(self, show_tracking: bool, distance_setup: bool, ret: bool = True) -> None:
        """Run the speed camera algorithm

        Args:
            show_tracking (bool): Flag indicating whether to display tracking information.
            distance_setup (bool): Flag indicating whether to setup distance measurement ratios.
            ret (bool, optional): Flag indicating if the stream is running or not. Defaults to True.
        """
        if self.input_mode == "burst_photos":
            file_names = sorted(os.listdir(self.input_path))
            for file_name in file_names:
                self.iter += 1
                image_path = os.path.join(self.input_path, file_name)
                frame = cv2.imread(image_path)
                if distance_setup and not self.distance_setup_ran:
                    clicks = []
                    fig, ax = plt.subplots(figsize=(16, 9))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ax.imshow(frame_rgb)
                    cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.speed_estimator.setup_distance(
                        event, frame_rgb, clicks, cid, fig))
                    plt.show()
                    self.distance_setup_ran = True
                self.process_frame(frame, show_tracking, distance_setup)
                self.io_handler.check_end_stream(ret)
            ret = False
            self.io_handler.check_end_stream(ret)

        elif self.input_mode == "video":
            while True:
                self.iter += 1
                ret, frame = self.io_handler.cap.read()
                if distance_setup and not self.distance_setup_ran:
                    fig, ax = plt.subplots(figsize=(16, 9))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ax.imshow(frame_rgb)
                    cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.speed_estimator.setup_distance(
                        event, frame_rgb, cid, fig))
                    plt.show()
                    self.distance_setup_ran = True
                self.process_frame(frame, show_tracking, distance_setup)
                self.io_handler.check_end_stream(ret)

        else:
            raise NotImplementedError

    def measure_fps(self, frame) -> None:
        """Measure and display the frames per second

        Args:
            frame (np.ndarray): The frame to display the FPS on.
        """
        curr_time = time.time()
        frame_time = curr_time-self.last_frame_time
        fps = int(1/frame_time)
        cv2.putText(frame, str(fps), [
                    frame.shape[1]-100, frame.shape[0]-50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.last_frame_time = time.time()


class VehicleDetector:
    def __init__(self):
        self.model_vd = YOLO('./YOLO_vehicle_detection/weights/yolov8s.pt')
        # ["bicycle", "car", "motorcycle", "bus", "truck"]
        self.searched_class_indices = [1, 2, 3, 5, 7]

    def detect_vehicles(self, frame: np.ndarray) -> np.ndarray:
        """Detect vehicles in a frame.

        Args:
            frame (np.ndarray): The frame in which to detect vehicles.

        Returns:
            np.ndarray: Detected vehicle bounding boxes. [x1,y1,x2,y2,conf]
        """
        detections = np.empty((0, 5))
        vd_results = self.model_vd(
            frame, stream=True, classes=self.searched_class_indices, conf=0.7, iou=0.9, agnostic_nms=True)
        for result in vd_results:
            boxes = result.boxes
            for box in boxes:
                x1_vd, y1_vd, x2_vd, y2_vd = tuple(map(int, box.xyxy[0]))
                cv2.rectangle(frame, (x1_vd, y1_vd),
                              (x2_vd, y2_vd), (255, 0, 255), 2)
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                cv2.putText(frame, f'{str(self.model_vd.model.names[cls])} {conf}', [
                            x1_vd, y1_vd], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                bbox_arr = np.array([x1_vd, y1_vd, x2_vd, y2_vd, conf])
                detections = np.vstack((detections, bbox_arr))
        return detections


class ObjectTracker:
    """Class for object tracking"""

    def __init__(self):
        self.tracker = Sort(max_age=60, min_hits=5, iou_threshold=0.4)
        self.vehicle_preds = {}

    def track_objects(self, frame: np.ndarray, detections: np.ndarray, show: bool) -> Dict[int, Dict]:
        """Track objects in a frame.

        Args:
            frame (np.ndarray): The frame to track objects in.
            detections (np.ndarray): Detected object bounding boxes.
            show (bool): Flag indicating whether to display tracking information.

        Returns:
            Dict[int, Dict]: Tracked object information.
        """
        track_results = self.tracker.update(detections)
        # [x1,y1,x2,y2,idx] - X coordinate from left, Y coordinate from top
        for vehicle in self.vehicle_preds:
            self.vehicle_preds[vehicle]['tracked'] = False

        for result in track_results:
            idx = int(result[-1])
            bboxes = result[:-1]
            bboxes_int = tuple(map(int, bboxes))
            # Prevent any minus values
            bboxes_int = np.clip(bboxes_int, a_min=0, a_max=None)
            center = (int((result[2]+result[0])/2),
                      int((result[3]+result[1])/2))

            if idx not in self.vehicle_preds:
                self.vehicle_preds[idx] = {}
                self.vehicle_preds[idx]['vd_center'] = []
                self.vehicle_preds[idx]['vd_bbox_coords'] = []
                self.vehicle_preds[idx]['lp_bbox_coords'] = []
                self.vehicle_preds[idx]['lp_conf'] = 0
                self.vehicle_preds[idx]['lp_text'] = ""
                self.vehicle_preds[idx]['ocr_conf'] = 0
                self.vehicle_preds[idx]['tracked'] = False
                self.vehicle_preds[idx]['tracking_window_opened'] = False
                self.vehicle_preds[idx]['speed'] = 0
                self.vehicle_preds[idx]['distance'] = []
                self.vehicle_preds[idx]['entering_time'] = 0
                self.vehicle_preds[idx]['leaving_time'] = 0
                self.vehicle_preds[idx]['direction'] = ""

            self.vehicle_preds[idx]['vd_bbox_coords'] = bboxes_int
            self.vehicle_preds[idx]['tracked'] = True

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
                cv2.putText(frame, f'{str(idx)}', [
                            bboxes_int[2], bboxes_int[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (B, G, R), 2)
            else:
                self.vehicle_preds[idx]['vd_center'] = center
                cv2.putText(frame, f'{str(idx)}', [
                            bboxes_int[2], bboxes_int[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return self.vehicle_preds


class LPEnhancer:
    """Class for license plate enchaning."""

    def __init__(self):
        self.rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Enhance the license plate image.

        Args:
            frame (np.ndarray): The license plate image to enhance.

        Returns:
            np.ndarray: The enhanced license plate image.
        """
        h, w = frame.shape[:2]
        test_lp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bil = cv2.bilateralFilter(test_lp_gray, 11, 17, 17)
        blackhat = cv2.morphologyEx(
            test_lp_gray, cv2.MORPH_BLACKHAT, self.rectKern)
        edges = cv2.Canny(bil, 10, 50, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

        angles = []
        try:
            for line in lines:
                rho, theta = line[0]
                angle = np.rad2deg(theta)
                angles.append(angle)
            predominant_angle = np.median(angles)
        except:
            predominant_angle = 90
        rotation_matrix = cv2.getRotationMatrix2D(
            (h / 2, w / 2), predominant_angle-90, 1)
        angle_corrected = cv2.warpAffine(blackhat, rotation_matrix, (w, h))

        enhanced_frame = angle_corrected

        return enhanced_frame


class OCR():
    """Class for optical character recognition"""

    def __init__(self):
        self.alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

    def read_license_plate(self, frame) -> Tuple[str, float]:
        """Reading license plate characters.

        Args:
            frame (np.ndarray): Image of the license plate.

        Returns:
            A tuple containing:
            - text(str): Detected license plate text.
            - conf(float): Confidence of the OCR.
        """
        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_LINE) as api:
            api.SetVariable('tessedit_char_whitelist', self.alphanumeric)
            image = Image.fromarray(frame)
            api.SetImage(image)
            text = api.GetUTF8Text()
            text = text.strip()
            confidences = api.AllWordConfidences()
            confidences = [float(c) for c in confidences]
            conf = np.mean(confidences) / 100
        return text, conf


class LicensePlateDetector:
    def __init__(self, oc_recognizer: OCR, image_enhancer: LPEnhancer):
        """
        Args:
            oc_recognizer(OCR): The object that performs license plate OCR.
            image_enhancer(LPEnhancer): The object that enhances license plate images.
        """
        self.model_lp = YOLO(
            './YOLO_license_plate_localization/weights/best.pt')
        self.oc_recognizer = oc_recognizer
        self.image_enhancer = image_enhancer

    def detect_license_plates(self, frame: np.ndarray, show_frame: np.ndarray, vehicle_preds: Dict) -> Dict:
        """Detect license plates in a frame and update vehicle predictions.

        Args:
            frame (np.ndarray): The frame to detect license plates in.
            show_frame (np.ndarray): The frame to display license plate information on.
            vehicle_preds (Dict): Tracked vehicle information.

        Returns:
            Dict: Updated vehicle predictions.
        """
        for idx in vehicle_preds.keys():
            bbox_vd = vehicle_preds[idx]['vd_bbox_coords']
            cropped_vehicle = np.array(
                frame[bbox_vd[1]:bbox_vd[3], bbox_vd[0]:bbox_vd[2]])

            if vehicle_preds[idx]['tracked'] == True:
                cv2.imshow(str(idx), cropped_vehicle)
                x_w, y_w, _, h_w = cv2.getWindowImageRect(str(idx))
                lp_text = vehicle_preds[idx]['lp_text']
                cv2.putText(show_frame, lp_text, [
                            bbox_vd[0], bbox_vd[3]+25], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                vehicle_preds[idx]['tracking_window_opened'] = True
            elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                cv2.destroyWindow(str(idx))
                if cv2.getWindowProperty(str(idx) + " license plate", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(str(idx) + " license plate")
                vehicle_preds[idx]['tracking_window_opened'] = False

            if vehicle_preds[idx]['tracked']:
                lp_preds = self.model_lp(cropped_vehicle, imgsz=320, iou=0.5)
                for result in lp_preds:
                    boxes = result.boxes
                    for box in boxes:
                        if len(box.xyxy[0]) != 0:
                            x1_lp, y1_lp, x2_lp, y2_lp = tuple(
                                map(int, box.xyxy[0]))
                            bbox_arr_lp = np.array(
                                [x1_lp, y1_lp, x2_lp, y2_lp])
                            lp_center = (
                                int(bbox_vd[0] + (x1_lp+x2_lp)/2), int(bbox_vd[1] + (y1_lp+y2_lp)/2))
                            vehicle_preds[idx]['lp_bbox_coords'] = bbox_arr_lp
                            cropped_lp = np.array(
                                cropped_vehicle[y1_lp:y2_lp, x1_lp:x2_lp])
                            conf = math.ceil((box.conf[0]*100))/100
                            vehicle_preds[idx]['lp_conf'] = conf

                            if vehicle_preds[idx]['tracked'] == True:
                                cv2.imshow(
                                    str(idx) + " license plate", cropped_lp)
                                cv2.moveWindow(
                                    str(idx) + " license plate", x_w, y_w + h_w)
                                vehicle_preds[idx]['tracking_window_opened'] = True
                            elif vehicle_preds[idx]['tracked'] == False and vehicle_preds[idx]['tracking_window_opened'] == True:
                                cv2.destroyWindow(str(idx) + " license plate")
                                vehicle_preds[idx]['tracking_window_opened'] = False

                            if lp_center[0] >= (frame.shape[1]*0.05):
                                enhanced_frame = self.image_enhancer.enhance_image(
                                    cropped_lp)
                                lp_text, ocr_conf = self.oc_recognizer.read_license_plate(
                                    enhanced_frame)
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


class SpeedEstimator:
    """Class for the speed estimation."""

    def __init__(self, fps: int):
        """
        Args:
            fps (int): Frames per second of the source file.
        """
        self.dist_in_meters = 0.0
        self.dist_in_pixels = 0.0
        self.pixel_meter_ratio = 0.0
        self.fps = fps
        self.moving_average_size = 3
        self.focal_length = 45/1000
        self.pixel_diameter = math.sqrt(17.3**2+13**2)/1000
        self.resolution_diameter = math.sqrt(1280**2+720**2)
        self.pixel_pitch = self.pixel_diameter/self.resolution_diameter
        self.car_real_diameter = math.sqrt(1.6**2+1.5**2)
        self.clicks = []
        self.dist_in_meters = 0.0

    def measure_distance(self, tracked_objects: Dict) -> Dict:
        """
        Measure the distance to tracked objects.

        Args:
            tracked_objects (Dict): Dictionary containing information about tracked objects.

        Returns:
            tracked_objects: Updated dictionary of tracked objects with distance measurements.
        """
        for idx in tracked_objects.keys():
            if tracked_objects[idx]['tracked']:
                x1, y1, x2, y2 = [tracked_objects[idx]
                                  ['vd_bbox_coords'][i] for i in range(4)]
                vehicle_diameter_in_pixel = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                # Distance to object = (Actual diameter of the object * Focal length) / diameter of the object in the image
                distance_in_meter = (self.car_real_diameter * self.focal_length) / \
                    (vehicle_diameter_in_pixel * self.pixel_pitch)
                tracked_objects[idx]['distance'].append(distance_in_meter)
        return tracked_objects

    def estimate_speed(self, frame: np.ndarray, tracked_objects: Dict, iter: int) -> Dict:
        """
        Estimate the speed of tracked objects.

        Args:
            frame(np.ndarray): Current frame of the video.
            tracked_objects (Dict): Dictionary containing information about tracked objects.
            iter(int): Current frame number.

        Returns:
            tracked_objects: Updated dictionary of tracked objects with speed estimates.
        """
        cv2.line(frame, (frame.shape[1], int(self.clicks[0][1])), (0, int(
            self.clicks[0][1])), (0, 0, 255), thickness=2)
        cv2.line(frame, (frame.shape[1], int(self.clicks[1][1])), (0, int(
            self.clicks[1][1])), (0, 0, 255), thickness=2)
        for idx in tracked_objects.keys():
            if len(tracked_objects[idx]['vd_center']) >= 2:
                box_diff = tracked_objects[idx]['vd_center'][-2][1] - \
                    tracked_objects[idx]['vd_center'][-1][1]
                if box_diff > 0:
                    tracked_objects[idx]['direction'] = "up"
                elif box_diff < 0:
                    tracked_objects[idx]['direction'] = "down"
                else:
                    tracked_objects[idx]['direction'] = "none"

                if tracked_objects[idx]['tracked'] and tracked_objects[idx]['direction'] == "up":
                    if tracked_objects[idx]['entering_time'] == 0 and tracked_objects[idx]['vd_center'][-1][1] < self.clicks[0][1] and tracked_objects[idx]['vd_center'][-1][1] > self.clicks[1][1]:
                        tracked_objects[idx]['entering_time'] = iter
                    if tracked_objects[idx]['leaving_time'] == 0 and tracked_objects[idx]['vd_center'][-1][1] < self.clicks[0][1] and tracked_objects[idx]['vd_center'][-1][1] < self.clicks[1][1]:
                        tracked_objects[idx]['leaving_time'] = iter
                    if tracked_objects[idx]['entering_time'] != 0 and tracked_objects[idx]['leaving_time'] != 0:
                        speed_in_frames = np.abs(
                            tracked_objects[idx]['entering_time'] - tracked_objects[idx]['leaving_time'])
                        speed = (self.dist_in_meters /
                                 ((1/self.fps) * speed_in_frames)) * 3.6
                        tracked_objects[idx]['speed'] = speed
                        cv2.putText(frame, str(int(speed)) + " km/h", (tracked_objects[idx]['vd_bbox_coords'][2],
                                    tracked_objects[idx]['vd_bbox_coords'][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if tracked_objects[idx]['tracked'] and tracked_objects[idx]['direction'] == "down":
                    if tracked_objects[idx]['leaving_time'] == 0 and tracked_objects[idx]['vd_center'][-1][1] > self.clicks[0][1] and tracked_objects[idx]['vd_center'][-1][1] > self.clicks[1][1]:
                        tracked_objects[idx]['leaving_time'] = iter
                    if tracked_objects[idx]['entering_time'] == 0 and tracked_objects[idx]['vd_center'][-1][1] < self.clicks[0][1] and tracked_objects[idx]['vd_center'][-1][1] > self.clicks[1][1]:
                        tracked_objects[idx]['entering_time'] = iter
                    if tracked_objects[idx]['entering_time'] != 0 and tracked_objects[idx]['leaving_time'] != 0:
                        speed_in_frames = np.abs(
                            tracked_objects[idx]['entering_time'] - tracked_objects[idx]['leaving_time'])
                        speed = (self.dist_in_meters /
                                 ((1/self.fps) * speed_in_frames)) * 3.6
                        tracked_objects[idx]['speed'] = speed
                        cv2.putText(frame, str(int(speed)) + " km/h", (tracked_objects[idx]['vd_bbox_coords'][2],
                                    tracked_objects[idx]['vd_bbox_coords'][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return tracked_objects

    def setup_distance(self, event, frame: np.ndarray, cid, fig: plt.figure) -> None:
        """
        Setup the distance measurement by selecting two points on the frame.

        Args:
            event: Event object from the Matplotlib figure.
            frame(np.ndarray): Current frame of the video.
            cid: Connection ID of the event handler.
            fig(plt.figure): Matplotlib figure.
        """
        if event.xdata is not None and event.ydata is not None:
            self.clicks.append((event.xdata, event.ydata))
            if len(self.clicks) == 2:
                p1, p2 = self.clicks
                self.dist_in_pixels = np.linalg.norm(
                    np.array(p1) - np.array(p2))
                fig.canvas.mpl_disconnect(cid)
                plt.imshow(frame)
                plt.scatter(*zip(*self.clicks), color='red', marker='x')
                plt.show()
                dist_in_meters = input(
                    "\nHow many meters is this in reality?\n")
                try:
                    self.dist_in_meters = float(dist_in_meters)
                    plt.close()
                except:
                    print("Not a number, stopping the program.")
                    sys.exit()


class IOHandler:
    """Class for input output handling."""

    def __init__(self, input_path: str):
        """
        Args:
            input_path (str): Path to the input video or photos.
        """
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path)

    def check_end_stream(self, ret: bool) -> None:
        """
        Check if the video stream has ended or if the user has quit.

        Args:
            ret (bool): Flag indicating if the video capture was successful.
        """
        if cv2.waitKey(100) & 0xFF == ord('q') or not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit()


def main():
    speed_camera = TrafficSpeedCamera(
        './samples/video_sample1.MOV', "video", fps=30)
    speed_camera.run(show_tracking=True, distance_setup=True)


if __name__ == '__main__':
    main()
