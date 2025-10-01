import pywhatkit
import pyttsx3
import datetime
import speech_recognition as sr
import webbrowser
from bs4 import BeautifulSoup
from time import sleep
import os
from datetime import datetime,timedelta
import pyautogui

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice",voices[1].id)
# engine.setProperty("rate",100)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def takeCommand():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source :
        print("listening.....")
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1
        recognizer.energy_threshold = 100
        audio = recognizer.listen(source,0,4)
    try:
        print("Understanding...")
        query = recognizer.recognize_google(audio,language="en-in")
        print(F"You said : {query}\n")
    except Exception as e:
        print("Say that again")
        return "None"
    return query

strTime = int(datetime.now().strftime("%H"))
update = int((datetime.now()+timedelta(minutes = 2)).strftime("%M"))

def sendmessage():
    speak("Who do you want to message?")
    a = int(input('''Person 1 - 1
    Person 2 - 2 '''))
    if a == 1:
        speak("What's the message?")
        message = str(input("Enter Message - "))
        pywhatkit.sendwhatmsg("+91..........", message, time_hour=strTime, time_min=update) #enter the number
        sleep(10)  # Sleep for a few seconds to ensure message typing completes
        pyautogui.press('enter')
        print("message send")
        speak("message send")
    elif a == 2:
        speak("What's the message?")
        message = str(input("Enter Message - "))
        pywhatkit.sendwhatmsg("+91..........", message, time_hour=strTime, time_min=update) #enter the number
        sleep(10)  # Sleep for a few seconds to ensure message typing completes
        pyautogui.press('enter')
        print("message send")
        speak("message send")