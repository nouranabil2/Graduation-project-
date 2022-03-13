
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
for x in range(3):
    ret, img = cap.read()
    #im = Image.open("img")
    cv2.imshow("img",img)
cv2.imwrite("img.png",img)
output=reader.readtext(img, detail=0)

for f in output:
        
        print(f)

key = cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()

speak(output)

