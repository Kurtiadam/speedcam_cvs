from ultralytics import YOLO

model_lp = YOLO('./YOLO_license_plate_localization/weights/best.pt')
input_path = r"C:\Users\Adam\Desktop\Test_set\License plate detection"
# for file_name in file_names:
    # image_path = os.path.join(input_path, file_name)
    # frame = cv2.imread(image_path)
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(100)
lp_results = model_lp(input_path,imgsz=320, conf = 0.5, iou=0.7, save_crop=True, save_txt = True, save_conf = True, save=True)