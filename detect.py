import cv2
import pytesseract

def get_roi(img, box):
    x, y, w, h = [int(x.item()) for x in box]
    y1 = int(y - .5*h)
    y2 = int(y + .5*h)
    x1 = int(x - .47*w)
    x2 = int(x + .47*w)
    roi = img[y1:y2, x1:x2]
    return roi

def preprocess_roi(roi):
    #roi = cv2.resize(roi, None, fx=.6, fy=.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    blur = cv2.GaussianBlur(noise, (5, 5), 0)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(norm, 128, 255, cv2.THRESH_BINARY)
    bill = cv2.bilateralFilter(thresh, 9, 75, 75)
    inv = 255 - bill
    cv2.imshow('inverted', inv)
    return inv

def get_text_from_image(inv):
    text = pytesseract.image_to_string(inv)
    return text

def process_detected_boxes(img, boxes):
    for box in boxes:
        roi = get_roi(img, box)
        processed_roi = preprocess_roi(roi)
        text = get_text_from_image(processed_roi)
        # print(f'Text in bounding box: {text}')
        return text

