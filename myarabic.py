import cv2
import datetime
import os
from gtts import gTTS
from playsound import playsound
import  speech_recognition as sr
import pyttsx3
from bidi.algorithm import get_display
import arabic_reshaper
from FaceR import FaceRecognition,extractEmbeddings
from predict import yolo_model

engine = pyttsx3.init('sapi5')
rate=engine.getProperty('rate')
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
	engine.say(audio)
	engine.runAndWait()
def formatArabicSentences(sentences):
   formatedSentences = arabic_reshaper.reshape(sentences)
   return get_display(formatedSentences)

def con(voice):
    try:
        obj=gTTS(text=voice,lang='ar',slow=False)
        obj.save('text.mp3')
        playsound('text.mp3')
        os.remove('text.mp3')
    except:
        return    
def command():
    r=sr.Recognizer()
    with sr.Microphone() as src:
     print('Say something....')
     audio=r.listen(src)
    try:
     t=r.recognize_google(audio,language='ar-AR')
     bid=formatArabicSentences(t)
     print(bid)
     #con("صباح الخير")
    #playsound('text.mp3')
    except sr.UnknownValueError as U:
     print(U)
     return
    except sr.RequestError as R:
     print(R)
     return
    return t 

def respond(query):
    # 1: greeting
    if 'مرحبا' in query:
        con("مرحبا نورا")
        return
    if 'كيف حالك'in query:
        con("بخير انت كيف حالك")
        return
    if  "سلام"in query:
        con("لى اللقاء ")
        exit() 

    if 'من حولي' in query:
    
          model = FaceRecognition()
          #cam = cv2.VideoCapture(0)
          count=0
          a=[]
          flag=0
          x=0
          #while True:
          #for x in range(3):
            
            #ret, image = cam.read()
          image=cv2.imread('download.jpg')
          names,boxes= model.recognize(image)
          model.draw_boxes(image,names,boxes)
            
            
          if x==0:
            con(" الاشخاص الذين تعرفهم حولك هم") 
            for f in names:
               if f != 'Unknown' :
                 flag=1

                 #speak(f)
                 con(f)
                 a.insert(count,f) 
                 count=count+1
          else:
             while(len(names)):
               for f in names:
                 for h in a:
                      if f == h:
                          names.remove(f)
                          break
                      

             for f in names:
                if f != 'Unknown' :
                    speak(f +"is around you")
                    a.insert(count,f) 
                    count=count+1
                #else:
                 # break
                


          if flag == 1:
                #speak("those are the people around you")
                cv2.imshow("image", image) 
                return
            #if cv2.waitKey(1) == 27:  # break if press ESC key
           
                   
    if'ماذا حولي'in query :
      yolo = yolo_model()
      #cam = cv2.VideoCapture(0)
      c =0 
      #for x in range(3):
       #     ret, image = cam.read()
      pred = yolo.predict(cv2.imread('data/dog.jpg'))
        #    pred = yolo.predict(image)
      
       
      for x in pred:
        if c!=0:
         speak("and"+pred[x][0][0] )  
        else:   
            c=1
            con("امامك الان")
            speak( "In front of you there is a"+pred[x][0][0])
          
      speak("That is the full view in front of you  ")   
      return

def wishMe():
	hour = int(datetime.datetime.now().hour)
	if hour>= 0 and hour<12:
		con("صباح الخير")
        

	else:
		con("مساء الخير")

wishMe()
while(1):
    voice_data=t=command()
    respond(voice_data)
#con(t)


