from ultralytics import YOLO
import os

model_vd = YOLO('./yolov8s.pt')
input_path = r"C:\Users\Adam\Desktop\Test_Dataset_2023.06.8"
searched_class_indices = [1, 2, 3, 5, 7]
# for file_name in file_names:
    # image_path = os.path.join(input_path, file_name)
    # frame = cv2.imread(image_path)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(100)
vd_results = model_vd(input_path, classes=searched_class_indices, conf=0.7, iou=0.9, save_crop=True, save_txt = True, save_conf = True, save=True)