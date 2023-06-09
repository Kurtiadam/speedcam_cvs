from ultralytics import YOLO

model_lp = YOLO('./YOLO_vehicle_detection/weights/yolov8s.pt')

if __name__ == '__main__':
    model_lp.train(data='./datasets/license_plate_detection_dataset/dataset_lp_local.yaml', epochs=5, imgsz=320)