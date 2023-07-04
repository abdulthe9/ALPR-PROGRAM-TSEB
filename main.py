from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np

# Load YOLOv8 model
plat = YOLO('runs/detect/train/weights/best.pt')

# Inisialisasi OCR engine
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Prediksi bounding boxes
img = cv2.imread('telloplat/drone.jpg')
detected = plat.predict(img, line_thickness=1, show=1)
tensor = detected[0].boxes.xywh

for box in tensor:
    x, y, w, h = [int(x.item()) for x in box]
    y1 = int(y - .5*h)
    y2 = int(y + .5*h)
    x1 = int(x - .47*w)
    x2 = int(x + .47*w)
    roi = img[y1:y2, x1:x2]

    # Extract the average color of the ROI
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    # Choose preprocessing based on the average color
    if avg_color[0] < 100 or avg_color[2] > 150:
        # Black or red license plate
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        blur = cv2.GaussianBlur(noise, (5, 5), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(norm, 128, 255, cv2.THRESH_BINARY)
        bill = cv2.bilateralFilter(thresh, 9, 75, 75)
        inv = 255 - bill
        processed_roi = inv
    else:
        # White or yellow license plate
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        blur = cv2.GaussianBlur(noise, (5, 5), 0)
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        bill = cv2.bilateralFilter(thresh, 9, 75, 75)
        processed_roi = bill

    #cv2.imshow('gray', gray)
    #cv2.imshow('noise', noise)
    #cv2.imshow('blur', blur)
    #cv2.imshow('normalize', norm)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('billaterall', bill)
    cv2.imshow('roi', processed_roi)
    text = pytesseract.image_to_string(processed_roi)
    print(f'Text in bounding box: {text}')

cv2.waitKey(0)


