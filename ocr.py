
from PIL import Image
import easyocr
import cv2
import pyttsx3

engine = pyttsx3.init('sapi5')
rate=engine.getProperty('rate')
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
def speak(audio):
	engine.say(audio)
	engine.runAndWait()

cap=cv2.VideoCapture(0)  
reader=easyocr.Reader(lang_list=['en'])
while True:
    ret, img = cap.read()
    im = Image.open("img")
    output=reader.readtext(im, detail=0)
#size=len(output)
    for f in output:
        speak(f)
        print(f)
    cap.release()
    cv2.destroyAllWindows() 

#read the image
#im = Image.open("4.jpg")


