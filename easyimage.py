
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

reader=easyocr.Reader(lang_list=['en'])
im = Image.open("b1.jpeg")
output=reader.readtext(im, detail=0)
#print(output)
#size=len(output)
for f in output:
        speak(f)
        print(f)
  




