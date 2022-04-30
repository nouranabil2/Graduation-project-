import cv2
import datetime
import os
from gtts import gTTS
from playsound import playsound
import  speech_recognition as sr
from bidi.algorithm import get_display
import arabic_reshaper
from FaceR import FaceRecognition,extractEmbeddings
from predict import yolo_model
from translate import Translator


def trans(word):
    translator= Translator(from_lang="english",to_lang="arabic")
    translation = translator.translate(word)
    return translation


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
   
    if 'كيف حالك'in query:
        con("بخير انت كيف حالك")
        return
   
   
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
                   # speak(f +"is around you")
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
            p=pred[x][0][0]
            translated=trans(p)
            con("و"+translated)
        else:   
            c=1
            con("امامك الان")
            p=pred[x][0][0]
            translated=trans(p)
            con(translated)
            
          
      #speak("That is the full view in front of you  ")   
      return

def wishMe():
	hour = int(datetime.datetime.now().hour)
	if hour>= 0 and hour<12:
		con("صباح الخير")
        

	else:
		con("مساء الخير")
        
def distance():
    return print("hi")
def fun(voice_data):
    while(1):
        voice_data=t=command()
        respond(voice_data)
        if voice_data  =="سلام":
            con("لى اللقاء ")
            break

    return
       

wishMe()
while(1):
    distance()
    voice_data=t=command()
    if voice_data == 'مرحبا':
        con("مرحبا نورا")
        fun(voice_data)
        

     

    
#con(t)


