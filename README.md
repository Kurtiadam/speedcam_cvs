# Speed camera algorithm with computer vision

This repository is for student work at the Budapest University of Technology and Economics' class of Computer Vision Systems.

![image](https://github.com/Kurtiadam/speedcam_cvs/assets/98428367/51f2c69d-5758-4fde-9175-7b40dd9e4655)


Throughout the semester we will develop a speed camera algorithm consisting of the following features:
- [x] Vehicle detection with YOLO
- [x] License plate localization
- [x] License plate reading with optical character recognition (OCR)
- [x] Vehicle speed estimation using optical flow

The students working on this project are:
- Pucsok Sándor (pucsoksandor)
- Kürti Ádám (Kurtiadam)

## How to use 
Required libraries and other resources:
- Python (3.8< and <3.11)
- libraries in requirements.txt
- tesserocr (download here: https://github.com/sirfz/tesserocr)

Run speedcam_cvs.py. 
In order to make use of the velocity measurement, a region of interest has to be set up which size in real life you have to know: Choose two points on the first frame of the source file in between which points you know the distance by two consecutive mouse clicks. Afterwards, specify this distance in meters in the console and the algorithm will start running.

**Please find the exhaustive documentation in the [_documentation_](https://github.com/Kurtiadam/speedcam_cvs/tree/main/documentation) folder.**
