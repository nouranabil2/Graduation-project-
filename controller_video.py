import random
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

   
    return t 

def respond(query):
   
    if 'كيف حالك'in query:
        con("بخير انت كيف حالك")
        return
   
   
    elif 'من حولي' in query:
    
        model = FaceRecognition()
        cam = cv2.VideoCapture(0)
        count=0
        a=[]
        flag=0
        x=0
          ##while True:
        #image=cv2.imread('2.jpg')
        for x in range(4):
          ret, image = cam.read()
          
          names,boxes= model.recognize(image)
          model.draw_boxes(image,names,boxes)  
          cv2.imshow('Input', image)
          c = cv2.waitKey(1)
          if c == 27:
            break
          if x==0:
            
            for f in names:
               if f != 'Unknown' :
                 
                 flag=1

                 #speak(f)
                 con(f)
                 a.insert(count,f) 
                 count=count+1
          else:
             #while(len(names)):
             for f in names:
                 for h in a:
                      if f == h:
                          names.remove(f)
                          break
                      

             for f in names:
                if f != 'Unknown' :
                   # speak(f +"is around you")
                    con(f)
                    a.insert(count,f) 
                    count=count+1
            
        if len(a):
            con("  هؤلاء هم الاشخاص الذين تعرفهم حولك ") 
        else:
            con(" لا يوجد حولك شخص تعرفه ")     
        
        return
            #if cv2.waitKey(1) == 27:  # break if press ESC key
           
                   
    if 'ماذا حولي' in query :
      yolo = yolo_model()
      cam = cv2.VideoCapture(0)
      count=0
      a=[]
      c =0 
      x=0
      for x in range(3):
            ret, image = cam.read()
      #pred = yolo.predict(cv2.imread('data/1.jpg'))
            pred = yolo.predict(image)     
            cv2.imshow('Input', image)
            c = cv2.waitKey(1)
            if c == 27:
                break   
            if x==0:   
                for f in pred:
                    con("امامك الان")
                    p=pred[x][0][0]
                    translated=trans(p)
                    con(translated)
                    con(f)
                    a.insert(count,f) 
                    count=count+1
            else:
             #while(len(names)):
             for f in pred:
                 for h in a:
                      if f == h:
                          #pred.remove(f)
                          del pred[f]
                          break
                      

             for f in pred:
               
                    p=pred[x][0][0]
                    translated=trans(p)
                    con("و"+translated)
                    con(f)
                    a.insert(count,f) 
                    count=count+1
            
       
      return     
          
      #speak("That is the full view in front of you  ")   
      return
    else:
        return


def wishMe():
	hour = int(datetime.datetime.now().hour)
	if hour>= 0 and hour<12:
		con("صباح الخير")
        

	else:
		con("مساء الخير")
        
def distance(x):
    return con(" احذر ")
def fun(voice_data):
    while(1):
            voice_data=t=command()
            if voice_data  =="سلام":
                con("إلى اللقاء ")
                break
            elif voice_data ==None:
                con("عذرا لا أستطيع سماعك")  
            else:
                respond(voice_data)



    return
       

wishMe()
while(1):
    x=random.uniform(0,10)
    print(x)
    if x<2:
        distance(x)
    voice_data=t=command()
    if voice_data == 'مرحبا':
        con("مرحبا نورا")
        fun(voice_data)
        

     

    
#con(t)


