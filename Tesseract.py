import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import os
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
def DocumentReading(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (3, 3), -1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 25, 18)
    myText = pytesseract.image_to_string(thresh, config='--psm 4')
    return myText
