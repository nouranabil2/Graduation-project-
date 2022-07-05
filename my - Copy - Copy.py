#from contextlib import nullcontext
#from sys import flags
from subprocess import call
import random
import playsound as ps # to play an audio file
import random
from time import ctime # get time details
import os
import numpy as np
import time

import cv2
import pyttsx3
import speech_recognition as sr
import datetime
from FaceR import FaceRecognition,extractEmbeddings
from predict import yolo_model
from gtts import gTTS
from playsound import playsound
import Tesseract as tr



def con(voice):
    try:
        obj=gTTS(text=voice,lang='en',slow=False)
        obj.save('text.mp3')
        playsound('text.mp3')
        os.remove('text.mp3')
    except:
        return        

def wishMe():
	hour = int(datetime.datetime.now().hour)
	if hour>= 0 and hour<12:
		con("Good Morning  !")

	elif hour>= 12 and hour<18:
		con("Good Afternoon  ! ")

	else:
		con("Good Evening  !")

	con("I am your Assistant. Welcome Noura.")



def takeCommand():
	#initialize the recognizer
	r = sr.Recognizer()
	#use microphone for input source
	with sr.Microphone() as source:
		
		print("Listening...")
		#r.pause_threshold = 1
        #listen for user
		audio=r.listen(source,timeout=8,phrase_time_limit=8)
	try:
		print("Recognizing...")
		query = r.recognize_google(audio, language ='en-in')
		print(f"User said: {query}\n")

	except Exception as e:
		#con("Sorry,Unable to Recognize your voice. can you please repeat again")
		return "None"
	
	return query
def there_exists(terms):
    for term in terms:
        if term in voice_data:
            return True
        

def respond(query):
    # 1: greeting
    if 'hi' in query:
        greet = "hello "
        con(greet)
        return
    if 'how are you'in query:
        con("I'm very well, thanks for asking ")
        return
    #2:time
    if there_exists(["what's the time","tell me the time","what time is it","what is the time"]):

        time = ctime().split(" ")[3].split(":")[0:2]
        if time[0] == "00":
            hours = '12'
        else:
            hours = time[0]
        minutes = time[1]
        time = f'{hours} {minutes}'
        con(time)    
        return

    if 'face' in query:
    #if there_exists(["Face","recignition","FaceRecognition"]):
          model = FaceRecognition()
          
          cam = cv2.VideoCapture(0)
          count=0
          a=[]
          flag=0
          #while True:
          for x in range(10):
            """
            if x==0:
                image=cv2.imread('4.jpg')
            if x ==1:
                image=cv2.imread('download.jpg')            
            if x ==3:
             image=cv2.imread('1.jpg')       
"""
            ret, image = cam.read()
           #image=cv2.imread('download.jpg')
            names,boxes= model.recognize(image)
            model.draw_boxes(image,names,boxes)
            cv2.imshow('Input', image)
            c = cv2.waitKey(1)
           
            print(names) 
            
            if x==0:
             for f in names:
               if f != 'Unknown' :
                 flag=1
                 con(f +"is around you")
                 a.insert(count,f) 
                 count=count+1
            else:
             for round1 in range(len(names)):
              for f in names:
                 if(f in a):   
                  names.remove(f)

                  

             for f in names:
                if f != 'Unknown' :
                    con(f +"is around you")
                    a.insert(count,f) 
                    names.remove(f)
                    count=count+1
                #else:
                 # break
                


          if flag == 1:
                con("those are the people around you")
                
                return
                   
    if'detection'in query :
      yolo = yolo_model()
      aobject=[]
      allo=[]
      allo_count=0
      count=0
      #cam = cv2.VideoCapture(0)
      k =0
      for round in range(5):
            allo=[]
            #ret, imageobj = cam.read()
        
            if round==0:
                imageobj=cv2.imread('data/1.jpg')
            if round ==1:
                imageobj=cv2.imread('data/6.jpg')            
            if round ==3:
                imageobj=cv2.imread('test.jpg')       

            
            pred = yolo.predict(imageobj)
           

            cv2.imshow('Input', imageobj)
            c = cv2.waitKey(1)

            # Create list of objects names in single frame
            for x in pred:
                allo.insert(allo_count,pred[x][0][0]) 
                allo_count =allo_count+1
                print(pred[x][0][0] )
               
                #if k!=0:
            #if round!=0:    
                    #while(len(allo)):
            for f in allo:
                for h in aobject:
                                if f == h:
                                    allo.remove(f)
                                    break
                    
            for new_k, new_val in pred.items():
                            print(new_k, len([item for item in new_val if item]))
                            if(new_val[0][0] in allo):
                             if (len(new_val)>1 ):
                                con("In front of you there are" )
                                con(str(len(new_val))+new_val[0][0])
                                aobject.insert(count,new_val[0][0]) 
                                count=count+1
                                allo.remove(new_val[0][0])

                             else:
                                con("and"+ new_val[0][0])
                                aobject.insert(count,new_val[0][0]) 
                                count=count+1
                        
         
          
                    
            
          
      con("That is the full view in front of you  ")   
      return
    if'document'in query :
        img=cv2.imread('dawa3.jpg')
        doc=tr.DocumentReading(img)
        #cv2.imshow('Input', img)
        #c = cv2.waitKey(1)
        print(doc)
        con(doc)
        con("That is the full document")
        return

     
    if there_exists(["exit", "quit", "goodbye","bye"]):
        con("bye Noura")
        exit()       
    else: con("Sorry,Unable to understand your command. can you please repeat again")


def distance(x):
    return con(" Warrning! In front of you there is an object ")
def fun(voice_data):
    while(1):
            voice_data="face"
            #voice_data=t=takeCommand().lower()
            if voice_data  =="bye":
                con("bye Noura")
                break
            elif voice_data ==None:
                con("Sorry,Unable to Recognize your voice. can you please repeat again")
            else:
                respond(voice_data)



    return
#wishMe()
while(1):
    x=random.uniform(0,5)
    print(x)
    if x<2:
        distance(x)
    voice_data="hey"    
    #voice_data=t=takeCommand().lower()
    if there_exists(["hey", "yafa","hey yafa"]):
        
        #con("hey noura, how can I help you ? ")
        fun(voice_data)
    if there_exists(["arabic", "language"]):
        
        #con("hey noura, how can I help you ? ")
        fun(voice_data)
            
        

"""""
wishMe() 
while(1):    
    #voice_data = query = takeCommand().lower()# get the voice input
    voice_data="detection"
    if voice_data != 'none' :
    
     x=respond(voice_data) # respond
"""""