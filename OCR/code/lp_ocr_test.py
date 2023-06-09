import pytesseract as pt
from pytesseract import Output
import cv2
import numpy as np
import easyocr
import time
import os
import sys
import pandas as pd

import tesserocr
from PIL import Image
api = tesserocr.PyTessBaseAPI()


def main():
    img_path = r"C:\Users\Adam\Desktop\speedcam_cvs\test_sets\License_plate_reading"
    file_names = os.listdir(img_path)
    df = pd.DataFrame(columns=['Prediction', 'Confidence'])
    for file_name in file_names:
        image_path = os.path.join(img_path, file_name)
        time_bef_crop = time.time()
        test_lp = cv2.imread(image_path)  

        w,h = test_lp.shape[:2]
        cv2.imshow("raw", test_lp)

        # Edge detection and correction
        test_lp_gray = cv2.cvtColor(test_lp, cv2.COLOR_BGR2GRAY)

        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(test_lp_gray, cv2.MORPH_BLACKHAT, rectKern)
        cv2.imshow('Blackhat', blackhat)

        edges = cv2.Canny(blackhat, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(blackhat, 1, np.pi / 180, 50)
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
                
        #         cv2.line(test_lp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow('Lines', test_lp)

        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta)
            angles.append(angle)
        predominant_angle = np.median(angles)
        print(predominant_angle)
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2),predominant_angle-90, 1)
        corrected_image = cv2.warpAffine(blackhat, rotation_matrix, (h, w))
        cv2.imshow('Corr', corrected_image)


        # Enhancing image
        # test_lp_gray_bw = cv2.threshold(corrected_image, 0, 255,cv2. THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # cv2.imshow('Gray', test_lp_gray_bw)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # eroded_image = cv2.erode(test_lp_gray_bw, kernel, iterations=1)
        # dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

        print("-------------------------------------------------------------")
        time_bef_crop = time.time()
        image = Image.fromarray(blackhat)
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_LINE) as api:
            api.SetVariable('tessedit_char_whitelist', alphanumeric)
            api.SetImage(image)
            text = api.GetUTF8Text()
            text = text.strip()
            confidences = api.AllWordConfidences()
            confidences = [float(c) for c in confidences]
            conf = np.mean(confidences) / 100
            print("TESSEROCR PRED: ", text, "CONF: ", conf)
            df.loc[len(df)] = [text, conf]

        time_aft_crop = time.time()
        print("TIME tesserocr: ", time_aft_crop-time_bef_crop, " sec\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()

    df.to_excel('./OCR/result/ocr_results.xlsx', index=False)


        # # pytesseract
        # time_bef_crop = time.time()
        # alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        # options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # options += " --psm {}".format(7)

        # data = pt.image_to_data(test_lp, config=options, output_type=Output.DICT)
        # confidences = data['conf']
        # confidences = [float(c) for c in confidences if c != '-1']
        # conf = np.mean(confidences) / 100
        # text = data['text']
        # text = ' '.join(text).strip()
        # print("PYTESSERACT PRED: ", text, "CONF: ", confidences)
        # time_aft_crop = time.time()
        # print("TIME PYTESSERACT: ", time_aft_crop-time_bef_crop, " sec\n")


        # # easyocr
        # time_bef_crop = time.time()
        # reader = easyocr.Reader(['en'])
        # image = img_path
        # result = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        # for bbox, text, confidence in result:
        #     print(f"EASYOCR PRED: {text}, CONF: {confidence}")
        # time_aft_crop = time.time()
        # print("TIME easyocr: ", time_aft_crop-time_bef_crop, " sec\n")

        # cv2.imshow("LP", test_lp)
        # cv2.waitKey(0)


if __name__ == "__main__":
    main()