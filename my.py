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

#import predict
engine = pyttsx3.init('sapi5')
rate=engine.getProperty('rate')
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def con(audio):
	engine.say(audio)
	engine.runAndWait()
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

	con("I am your Assistant. Welcome Noura. How can i Help you?")



def takeCommand():
	#initialize the recognizer
	r = sr.Recognizer()
	#use microphone for input source
	with sr.Microphone() as source:
		
		print("Listening...")
		#r.pause_threshold = 1
        #listen for user
		audio = r.listen(source,)
	try:
		print("Recognizing...")
		query = r.recognize_google(audio, language ='en-in')
		print(f"User said: {query}\n")

	except Exception as e:
		print(e)
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
    #3:music
    if 'play music' in query or "play song" in query:
            con("Here you go with music")
            # music_dir = "G:\\Song"
            music_dir = "C:\\Users\\Dell\\Music"
            songs = os.listdir(music_dir)
            #print(songs)   
            random = os.startfile(os.path.join(music_dir, songs[1]))     
            return
    if 'face' in query:
    #if there_exists(["Face","recignition","FaceRecognition"]):
          model = FaceRecognition()
          
          #cam = cv2.VideoCapture(0)
          count=0
          a=[]
          flag=0
          #while True:
          for x in range(3):
            
           # ret, image = cam.read()
           image=cv2.imread('download.jpg')
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
             while(len(names)):
               for f in names:
                 for h in a:
                      if f == h:
                          names.remove(f)
                          break
                      

             for f in names:
                if f != 'Unknown' :
                    con(f +"is around you")
                    a.insert(count,f) 
                    count=count+1
                #else:
                 # break
                


          if flag == 1:
                con("those are the people around you")
                
                return
           
           
                   
    if'detection'in query :
      yolo = yolo_model()
      #cam = cv2.VideoCapture(0)
      k =0
      #for x in range(3):
       #     ret, image = cam.read()
      imageobj=cv2.imread('data/dog.jpg')
      pred = yolo.predict(imageobj)
        #    pred = yolo.predict(image)
        
      cv2.imshow('Input', imageobj)
      c = cv2.waitKey(1)
    
       
      for x in pred:
        print(pred[x][0][0] )
        if k!=0:
         con("and"+pred[x][0][0] )  
         
        else:   
            k=1
            #con( "In front of you there is a"+ pred[x][0][0])
            con("In front of you there is a" )
            con(pred[x][0][0])
            
          
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
    else: con("Sorry,Unable to Recognize your voice. can you please repeat again")


wishMe() 
while(1):    
    voice_data = query = takeCommand().lower()# get the voice input
    #voice_data="face"
    if voice_data != 'none' :
    
     x=respond(voice_data) # respond
