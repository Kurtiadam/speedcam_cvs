from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load a pretrained model


if __name__ == '__main__':
    # Train the model
    model.train(data='./datasets/license_plate_detection_dataset/dataset_lp_local.yaml', epochs=5, imgsz=640)