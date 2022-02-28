import datetime
from sys import flags
import pyttsx3
import speech_recognition as sr
import random
import playsound # to play an audio file
import random
from time import ctime # get time details
import os
import numpy as np
import time
import cv2
from subprocess import call
from FaceR import FaceRecognition,extractEmbeddings
from predict import yolo_model
#import predict

     
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def speak(audio):
	engine.say(audio)
	engine.runAndWait()

def wishMe():
	hour = int(datetime.datetime.now().hour)
	if hour>= 0 and hour<12:
		speak("Good Morning  !")

	elif hour>= 12 and hour<18:
		speak("Good Afternoon  !")

	else:
		speak("Good Evening  !")

	assname =("Zomba")
	speak("I am your Assistant")
	speak(assname)
	

def username():
	speak("Welcome Noura. How can i Help you?")
	

def takeCommand():
	
	r = sr.Recognizer()
	
	with sr.Microphone() as source:
		
		print("Listening...")
		r.pause_threshold = 1
		audio = r.listen(source)

	try:
		print("Recognizing...")
		query = r.recognize_google(audio, language ='en-in')
		print(f"User said: {query}\n")

	except Exception as e:
		print(e)
		#speak("Sorry,Unable to Recognize your voice. can you please repeat again")
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
        speak(greet)
    if 'how are you'in query:
        speak("I'm very well, thanks for asking ")
    #2:time
    if there_exists(["what's the time","tell me the time","what time is it","what is the time"]):

        time = ctime().split(" ")[3].split(":")[0:2]
        if time[0] == "00":
            hours = '12'
        else:
            hours = time[0]
        minutes = time[1]
        time = f'{hours} {minutes}'
        speak(time)    
    #3:music
    if 'play music' in query or "play song" in query:
            speak("Here you go with music")
            # music_dir = "G:\\Song"
            music_dir = "C:\\Users\\Dell\\Music"
            songs = os.listdir(music_dir)
            print(songs)   
            random = os.startfile(os.path.join(music_dir, songs[1]))     
    if'close'in query :
            songs.close()   
    if'recognition'in query :
          model = FaceRecognition()
          #cam = cv2.VideoCapture(0)
    
          extractEmbeddings()
          count=0
          flag=0
          while True:
           
            #ret, image = cam.read()
            image=cv2.imread('download.jpg')
            names,boxes= model.recognize(image)
            model.draw_boxes(image,names,boxes)
            for f in names:
             if f != 'Unknown':
                 flag=1
                 speak(f +"is around you")
            cv2.imshow("image", image) 
            
            #if cv2.waitKey(1) == 27:  # break if press ESC key
            if flag==0:
                
                speak("all people around you are unknown")
                speak("goodbye Noura")
                exit()

                
            if flag == 1:
             speak("Those are the known people around you  ")   
             speak("goodbye Noura")
             exit()   
                   
    if'view'in query :
      yolo = yolo_model()
      c =0 
      pred = yolo.predict(cv2.imread('data/dog.jpg'))
       
      for x in pred:
           if c!=0:
              speak("and" )  
           c=pred[x][0][5]*100
           intc=int(c)
           speak("  by" )  
           speak(intc ) 
           speak( "Percent in front of you there is a"+pred[x][0][0])
      speak("That is the full view in front of you  ")   
      speak("goodbye Noura")
      exit()
    if there_exists(["exit", "quit", "goodbye","bye"]):
        speak("bye Noura")
        exit()       
    #else: speak("Sorry,Unable to Recognize your voice. can you please repeat again")

wishMe() 
username() 
  
while(1):   
    voice_data = query = takeCommand().lower()# get the voice input
    if voice_data != 'none' :
     x=respond(voice_data) # respond
     if x == None:
       speak("Sorry,Unable to Recognize your voice. can you please repeat again")
    
